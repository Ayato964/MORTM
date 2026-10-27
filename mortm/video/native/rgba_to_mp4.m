#import <AVFoundation/AVFoundation.h>
#import <Accelerate/Accelerate.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static size_t read_exact(int fd, uint8_t *buf, size_t count) {
    size_t total = 0;
    while (total < count) {
        ssize_t n = read(fd, buf + total, count - total);
        if (n <= 0) {
            break;
        }
        total += (size_t)n;
    }
    return total;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 5) {
            fprintf(stderr, "Usage: rgba_to_mp4 <output.mp4> <even_width> <even_height> <fps> [bitrate_bps]\n");
            return 1;
        }

        NSString *outputPath = [NSString stringWithUTF8String:argv[1]];
        int width = atoi(argv[2]);
        int height = atoi(argv[3]);
        int fps = atoi(argv[4]);
        int bitrate = (argc >= 6) ? atoi(argv[5]) : 14000000;

        if (width <= 0 || height <= 0 || fps <= 0 || (width % 2) != 0 || (height % 2) != 0) {
            fprintf(stderr, "[rgba_to_mp4] Width and height must be positive even integers, fps > 0.\n");
            return 1;
        }

        NSURL *outputURL = [NSURL fileURLWithPath:outputPath];
        NSFileManager *fm = [NSFileManager defaultManager];
        if ([fm fileExistsAtPath:outputPath]) {
            NSError *rmErr = nil;
            [fm removeItemAtURL:outputURL error:&rmErr];
        }

        NSError *writerErr = nil;
        AVAssetWriter *writer = [[AVAssetWriter alloc] initWithURL:outputURL
                                                          fileType:AVFileTypeMPEG4
                                                             error:&writerErr];
        if (!writer || writerErr) {
            fprintf(stderr, "[rgba_to_mp4] Failed to create AVAssetWriter: %s\n",
                    writerErr ? [[writerErr localizedDescription] UTF8String] : "unknown");
            return 1;
        }

        NSDictionary *compressionProps = @{
            AVVideoAverageBitRateKey : @(bitrate),
            AVVideoExpectedSourceFrameRateKey : @(fps),
            AVVideoMaxKeyFrameIntervalKey : @(fps * 2),
            AVVideoProfileLevelKey : AVVideoProfileLevelH264HighAutoLevel,
        };
        NSDictionary *videoSettings = @{
            AVVideoCodecKey : AVVideoCodecTypeH264,
            AVVideoWidthKey : @(width),
            AVVideoHeightKey : @(height),
            AVVideoCompressionPropertiesKey : compressionProps,
        };

        AVAssetWriterInput *writerInput = [AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo
                                                                             outputSettings:videoSettings];
        writerInput.expectsMediaDataInRealTime = NO;

        NSDictionary *pixelBufferAttrs = @{
            (NSString *)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA),
            (NSString *)kCVPixelBufferWidthKey : @(width),
            (NSString *)kCVPixelBufferHeightKey : @(height),
            (NSString *)kCVPixelBufferCGImageCompatibilityKey : @YES,
            (NSString *)kCVPixelBufferCGBitmapContextCompatibilityKey : @YES,
        };

        AVAssetWriterInputPixelBufferAdaptor *adaptor =
            [AVAssetWriterInputPixelBufferAdaptor assetWriterInputPixelBufferAdaptorWithAssetWriterInput:writerInput
                                                                             sourcePixelBufferAttributes:pixelBufferAttrs];

        if (![writer canAddInput:writerInput]) {
            fprintf(stderr, "[rgba_to_mp4] Cannot add H.264 input to AVAssetWriter.\n");
            return 1;
        }
        [writer addInput:writerInput];

        if (![writer startWriting]) {
            fprintf(stderr, "[rgba_to_mp4] startWriting failed: %s\n",
                    writer.error ? [[writer.error localizedDescription] UTF8String] : "unknown");
            return 1;
        }
        [writer startSessionAtSourceTime:kCMTimeZero];

        size_t frameBytes = (size_t)width * (size_t)height * 4;
        uint8_t *rawRgba = (uint8_t *)malloc(frameBytes);
        if (!rawRgba) {
            fprintf(stderr, "[rgba_to_mp4] Out of memory allocating frame buffer.\n");
            return 1;
        }

        const uint8_t permuteMap[4] = {2, 1, 0, 3}; // RGBA -> BGRA
        int64_t frameIdx = 0;

        while (1) {
            @autoreleasepool {
                size_t got = read_exact(STDIN_FILENO, rawRgba, frameBytes);
                if (got == 0) {
                    break;
                }
                if (got < frameBytes) {
                    fprintf(stderr, "[rgba_to_mp4] Ignoring partial trailing frame (%zu / %zu bytes).\n", got, frameBytes);
                    break;
                }

                while (!writerInput.readyForMoreMediaData) {
                    usleep(2000);
                }

                CVPixelBufferPoolRef pool = adaptor.pixelBufferPool;
                if (!pool) {
                    fprintf(stderr, "[rgba_to_mp4] PixelBufferPool is NULL.\n");
                    free(rawRgba);
                    return 1;
                }

                CVPixelBufferRef pxBuf = NULL;
                CVReturn status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &pxBuf);
                if (status != kCVReturnSuccess || !pxBuf) {
                    fprintf(stderr, "[rgba_to_mp4] CVPixelBufferPoolCreatePixelBuffer failed (%d).\n", status);
                    free(rawRgba);
                    return 1;
                }

                CVPixelBufferLockBaseAddress(pxBuf, 0);
                void *dstBase = CVPixelBufferGetBaseAddress(pxBuf);
                size_t dstRowBytes = CVPixelBufferGetBytesPerRow(pxBuf);

                vImage_Buffer src = {
                    .data = rawRgba,
                    .height = (vImagePixelCount)height,
                    .width = (vImagePixelCount)width,
                    .rowBytes = (size_t)width * 4,
                };
                vImage_Buffer dst = {
                    .data = dstBase,
                    .height = (vImagePixelCount)height,
                    .width = (vImagePixelCount)width,
                    .rowBytes = dstRowBytes,
                };
                vImagePermuteChannels_ARGB8888(&src, &dst, permuteMap, kvImageNoFlags);
                CVPixelBufferUnlockBaseAddress(pxBuf, 0);

                CMTime pts = CMTimeMake(frameIdx, fps);
                BOOL ok = [adaptor appendPixelBuffer:pxBuf withPresentationTime:pts];
                CVPixelBufferRelease(pxBuf);

                if (!ok) {
                    fprintf(stderr, "[rgba_to_mp4] appendPixelBuffer failed at frame %lld: %s\n",
                            frameIdx,
                            writer.error ? [[writer.error localizedDescription] UTF8String] : "unknown");
                    free(rawRgba);
                    return 1;
                }
                frameIdx++;
            }
        }

        free(rawRgba);
        [writerInput markAsFinished];

        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        [writer finishWritingWithCompletionHandler:^{
            dispatch_semaphore_signal(sem);
        }];
        dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);

        if (writer.status != AVAssetWriterStatusCompleted) {
            fprintf(stderr, "[rgba_to_mp4] finishWriting failed: %s\n",
                    writer.error ? [[writer.error localizedDescription] UTF8String] : "unknown");
            return 1;
        }
    }
    return 0;
}

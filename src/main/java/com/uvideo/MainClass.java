package com.uvideo;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Stream;

import net.coobird.thumbnailator.Thumbnails;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacv.*;

import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.opencv.opencv_java;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.BackgroundSubtractor;
import org.opencv.video.Video;

import javax.imageio.ImageIO;

import static com.uvideo.ProcessPixelLine.DIFF;
import static org.bytedeco.ffmpeg.global.avcodec.AV_CODEC_ID_VP9;
import static org.opencv.core.CvType.*;
import static org.opencv.imgproc.Imgproc.*;


public class MainClass {

    /**
     * all distances are specified in pixels
     * any colors are represented as single-channel gray images 0..255
     * FLUCTUATIONS_HEIGHT - percentage of deviation from the height when building
     * a symbolic image. the text output will be different, but the output images
     * will be fixed size
     * BACK_SUB - removing a still background, most likely you don't want to use it.
     * BLACK - black background
     * LINE_SPACING - line spacing when capturing and outputting
     * FILL_DEPTH - lower limit for filling the image if the symbol is not
     * represented as a contour.
     * FILL_ALIGNMENT - aligns the fill relative to the absolute position, regardless
     * of the previous characters. do not use it with monospaced fonts and when
     * displaying images in text, as the quality will be lost.
     * SPLIT_FILL - splits the outline and fill into different images.
     * SPIN - rotation of characters if they are not marked as _dont_spin and _dont_move.
     * USE_THRESH - applying an openCV adaptive threshold to the original image. you
     * should only disable it if the image is already a binary image of the contours.
     * note that MIN_WEIGHT + DIFF is added to it.
     * BETTER_THRESH - removed most of the single and double pixels. work if USE_2_THRESH = false
     * USE_2_THRESH - using the second adaptive threshold on top of the first
     * SYMBOLS_FOLDER - when output to text, the folder must contain chars.txt with
     * a sequential enumeration of all characters. changing the order of characters by
     * renaming results in incorrect operation.
     * INPUT_FILE_NAME - name of the file in the input&output directory. can be passed
     * as an argument to MainClass, with any full path.
     */

    private static final int     HEIGHT = 480;
    private static final double  FLUCTUATIONS_HEIGHT = 0.;
    private static final int     FRAMERATE = 0;        // 0 as source
    private static final int     CREATE_FRAMES = 0;    // 0 all
    private static final int     SKIPPED_FRAMES = 0;
    private static final boolean BACK_SUB = false;
    public  static final boolean BLACK = true;
    public  static final int     LINE_SPACING = 0;
    public  static final double  FILL_DEPTH = 100.;
    public  static final boolean FILL_ALIGNMENT = true;
    public  static final boolean SPLIT_FILL = false;
    public  static final boolean SPIN = true;
    private static final boolean USE_CANNY = false;
    private static final boolean USE_THRESH = true;
    private static final boolean USE_2_THRESH = false;
    private static final boolean BETTER_THRESH = false;
    public  static final String  PATCH;
    public  static final String  SYMBOLS_FOLDER = "MS_Gothic.ttf_14_00";
    private static       String  INPUT_FILE_NAME = "sample.webm"; // can be a command line parameter
    public  static final boolean OUTPUT_FRAMES = true;
    private static final boolean OUTPUT_VIDEO = true;
    private static final boolean OUTPUT_CANNY = true;
    private static final boolean OUTPUT_THRESH = true;
    private static final boolean OUTPUT_TEXT = true;
    private static final boolean OUTPUT_ORIGINAL_FRAMES = false;
    public  static       int     SYMBOL_HEIGHT; // automatic
    private static final List<File> symbols;
    private static final List<Integer> codePoints;
    private static final Pair<Integer, Integer> THRESH_COEFFICIENTS;

    static {
        // the first value cannot be even
        THRESH_COEFFICIENTS = new Pair<>(3, 3); // 5, 13
        System.out.println("availableProcessors " + Runtime.getRuntime().availableProcessors());
        PATCH = new File("").getAbsolutePath() + "\\data_set\\";
        File folder = new File(PATCH);
        if (!folder.exists())
            throw new RuntimeException("Folder " + folder.getAbsolutePath() + " not found");
        folder = new File(PATCH + SYMBOLS_FOLDER);
        if (!folder.exists())
            throw new RuntimeException("Folder " + folder.getAbsolutePath() + " not found");
        folder = new File(PATCH + "frames");
        if (!folder.exists()) folder.mkdir();
        folder = new File(PATCH + "canny");
        if (!folder.exists()) folder.mkdir();
        folder = new File(PATCH + "thresh");
        if (!folder.exists()) folder.mkdir();
        folder = new File(PATCH + "mask");
        if (!folder.exists()) folder.mkdir();
        folder = new File(PATCH + "rotated_symbols");
        if (!folder.exists()) folder.mkdir();
        folder = new File(PATCH + "fill");
        if (!folder.exists()) folder.mkdir();
        folder = new File(PATCH + "text");
        if (!folder.exists()) folder.mkdir();
        folder = new File(PATCH + "input_frames");
        if (!folder.exists()) folder.mkdir();
        System.out.println("Start loading OpenCV Java native library...");
        Long time = System.currentTimeMillis();
        Loader.load(opencv_java.class);
        time = System.currentTimeMillis() - time;
        System.out.println("Loaded in " + TimeUnit.MILLISECONDS.toSeconds(time) + "s");
        time = System.currentTimeMillis();

        symbols = new ArrayList<>(100);
        try (Stream<Path> paths = Files.walk(Paths
                .get(PATCH + SYMBOLS_FOLDER))) {
            paths
                    .filter(Files::isRegularFile)
                    .map(Path::toFile)
                    .filter(f -> f.getName()
                            .substring(f.getName()
                                    .lastIndexOf(".") + 1)
                            .equals("png"))
                    .sorted()
                    .forEach(symbols::add);
        } catch (IOException e) {
            e.printStackTrace();
        }

        codePoints = new ArrayList<>(symbols.size());
        try (Scanner sc = new Scanner(new File(PATCH + SYMBOLS_FOLDER + "\\chars.txt"))) {
            while (sc.hasNext()) {
                String str = sc.nextLine();
                if (!str.isEmpty()) codePoints.add(str.codePointAt(str.length() - 1));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        try {
            if (codePoints.isEmpty()) Logger.getGlobal().log(Level.INFO, "chars.size() == 0");
            else if (symbols.size() != codePoints.size())
                throw new IllegalArgumentException("symbols.size() = " + symbols.size()
                        + " != chars.size() = " + codePoints.size());
            SYMBOL_HEIGHT = ProcessPixelLine.setSymbols(symbols, codePoints);
        } catch (IllegalArgumentException e) {
            e.printStackTrace();
            System.exit(-1);
        }

        time = System.currentTimeMillis() - time;
        System.out.println("Loaded images " + TimeUnit.MILLISECONDS.toSeconds(time) + "s");
    }

    public static String getInputFileName() {
        return INPUT_FILE_NAME;
    }

    private static void writeLinesToFile(String[] lines, String filename) {
        if (!codePoints.isEmpty()) {
            File file = new File(PATCH + "text\\" + filename);
            try (PrintWriter writer = new PrintWriter(file, StandardCharsets.UTF_8)) {
                for (var line : lines)
                    writer.println(line);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private static Pair<Mat, Mat> createUtf8Mat(Mat threshImg, Mat grayImg, Mat thresh2Img, int fNumber) {
        final int numberOfRows = grayImg.rows() / (SYMBOL_HEIGHT + LINE_SPACING);
        CountDownLatch cdl = new CountDownLatch(numberOfRows);
        int nOfThreads = Runtime.getRuntime().availableProcessors() - 1;
        ArrayList<ProcessLine<Mat>> lines = new ArrayList<>(numberOfRows);
        ExecutorService executor = Executors.newFixedThreadPool(nOfThreads);

        // sending the lines for processing
        for (int i = 0; i < numberOfRows; i++) {
            Mat threshLine = threshImg.submat(new Rect(0,
                    i * (SYMBOL_HEIGHT + LINE_SPACING), threshImg.cols(), SYMBOL_HEIGHT));
            Mat grayLine = grayImg.submat(new Rect(0,
                    i * (SYMBOL_HEIGHT + LINE_SPACING), threshImg.cols(), SYMBOL_HEIGHT));
            if (thresh2Img != null) {
                Mat thresh2Line = thresh2Img.submat(new Rect(0,
                        i * (SYMBOL_HEIGHT + LINE_SPACING), threshImg.cols(), SYMBOL_HEIGHT));
                lines.add(new ProcessPixelLine(threshLine, grayLine, thresh2Line, cdl));
            } else lines.add(new ProcessPixelLine(threshLine, grayLine, fNumber, i + 1, cdl));
            executor.execute(lines.get(i));
        }

        try {
            cdl.await();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        executor.shutdown();

        Mat fin = new Mat(threshImg.rows(), threshImg.cols(), CV_8UC1, new Scalar(0.));
        Mat fill = new Mat(threshImg.rows(), threshImg.cols(), CV_8UC1, new Scalar(BLACK ? 0. : 255.));
        String[] textFin = new String[numberOfRows];
        for (int i = 0; i < numberOfRows; i++) {
            Mat resultLine = lines.get(i).getResult();
            resultLine.copyTo(fin.submat(new Rect(0,
                    i * (SYMBOL_HEIGHT + LINE_SPACING), threshImg.cols(), SYMBOL_HEIGHT)));
            // creates a notebook effect if the distance between the lines is greater than 2
            if (!BLACK) new Mat(LINE_SPACING / 2, threshImg.cols(), CV_8UC1, new Scalar(255.)).copyTo(fin.
                    submat(new Rect(0, i * (SYMBOL_HEIGHT + LINE_SPACING) + SYMBOL_HEIGHT, threshImg.cols(), LINE_SPACING / 2)));
            Mat fillLine = lines.get(i).getFill();
            fillLine.copyTo(fill.submat(new Rect(0, i * (SYMBOL_HEIGHT + LINE_SPACING), threshImg.cols(), SYMBOL_HEIGHT)));
            if (!codePoints.isEmpty()) textFin[i] = lines.get(i).getTextResult();
        }

        if (OUTPUT_TEXT)
            writeLinesToFile(textFin, String.format("%s-%03d.txt", "text", fNumber));

        return new Pair<>(fin, fill);
    }

    public static BufferedImage resize(BufferedImage img, int newW, int newH) throws IOException {
        return Thumbnails.of(img).forceSize(newW, newH).asBufferedImage();
    }

    private static void blurFineLines(Mat src, Mat dst) {
        Imgproc.GaussianBlur(src, dst, new Size(3, 3), 0);
        /*for (int i = 0; i < dst.rows(); i++) {
            for (int j = 0; j < dst.cols(); j++) {
                double p = dst.get(i, j)[0];
                if (p < 255.) {
                    dst.put(i, j, p - 20.);
                }
            }
        }*/
    }

    public static void main(String[] args) {
        String fileName;
        if (args.length > 0) {
            fileName = args[0].contains("\\") ? args[0] : PATCH + "input&output\\" + args[0];
            INPUT_FILE_NAME = fileName.substring(fileName.lastIndexOf("\\") + 1);
        } else fileName = PATCH + "input&output\\" + INPUT_FILE_NAME;
        String dstName = fileName.replace(fileName.
                substring(fileName.lastIndexOf(".")), "_converted.webm");

        try (FFmpegFrameGrabber g = new FFmpegFrameGrabber(fileName)) {
            if (FRAMERATE > 0) g.setFrameRate(FRAMERATE);
            g.start();
            if (HEIGHT >= SYMBOL_HEIGHT + LINE_SPACING) {
                g.setImageWidth((int) ((double) HEIGHT / g.getImageHeight() * g.getImageWidth()));
                g.setImageHeight(HEIGHT);
            }
            final int totalCreateVFrames = g.getLengthInVideoFrames() - SKIPPED_FRAMES;

            try (FFmpegFrameRecorder recorder = new FFmpegFrameRecorder(dstName, g.getImageWidth(), g.getImageHeight(), g.getAudioChannels())) {
                if (OUTPUT_VIDEO) {
                    recorder.setFrameRate(g.getFrameRate());
                    //recorder.setSampleFormat(g.getSampleFormat());
                    recorder.setSampleRate(48000);
                    recorder.setAudioMetadata(g.getAudioMetadata());
                    recorder.setVideoMetadata(g.getVideoMetadata());
                    recorder.setVideoCodec(AV_CODEC_ID_VP9);
                    recorder.setVideoBitrate(2500000);
                    recorder.setAudioBitrate(128000);
                    /*HashMap<String, String> options = new HashMap<>();
                    options.put("codec:v", "libvpx-vp9");
                    options.put("pix_fmt", "yuv420p");
                    options.put("vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2");
                    options.put("b:v", "1500k");
                    options.put("codec:a", "libopus");
                    options.put("b:a", "128k");
                    options.put("ar", "48000");
                    recorder.setOptions(options);*/
                    recorder.setFormat("webm");
                    recorder.start();
                }

                BackgroundSubtractor backSub;
                Mat fgMask;
                if (BACK_SUB) {
                    fgMask = new Mat(g.getImageHeight(), g.getImageWidth(), CV_8UC1);
                    backSub = Video.createBackgroundSubtractorKNN(1, 50, false);
                    //backSub = Video.createBackgroundSubtractorMOG2(1, 50, false);
                }

                Java2DFrameConverter java2dFrameConverter = new Java2DFrameConverter();
                OpenCVFrameConverter.ToOrgOpenCvCoreMat converter = new OpenCVFrameConverter.ToOrgOpenCvCoreMat();
                Frame fr;
                int vFrNumber = 0, createdVFrNumber = 0;
                long startTimeMillis = System.currentTimeMillis(), timestamp;

                while ((fr = g.grab()) != null) {
                    timestamp = fr.timestamp;

                    if (fr.image != null) {
                        vFrNumber++;
                        if (vFrNumber <= SKIPPED_FRAMES) continue;
                        createdVFrNumber++;

                        if (OUTPUT_ORIGINAL_FRAMES) {
                            ImageIO.write(java2dFrameConverter.convert(fr), "png",
                                    new File(String.format(PATCH + "input_frames\\frame-%03d.png", vFrNumber)));
                        }

                        if (FLUCTUATIONS_HEIGHT != 0.) {
                            int shift = vFrNumber % 72;
                            int frameHeight = (int) (HEIGHT + HEIGHT / 18. * FLUCTUATIONS_HEIGHT * (18. - (shift > 36 ? 72 - shift : shift)));
                            //System.out.println("frameHeight: " + frameHeight);
                            int frameWidth = (int) ((double) frameHeight / fr.imageHeight * fr.imageWidth);
                            fr = java2dFrameConverter.convert(resize(java2dFrameConverter.convert(fr), frameWidth, frameHeight));
                        }

                        Mat grabbedImage = converter.convert(fr);
                        int rows = grabbedImage.rows(), cols = grabbedImage.cols();

                        if (BACK_SUB) backSub.apply(grabbedImage, fgMask, 0.01);

                        Mat gray = new Mat(rows, cols, COLOR_BGR2GRAY);
                        Imgproc.cvtColor(grabbedImage, gray, COLOR_BGR2GRAY);

                        Mat thresh, thresh2 = null;
                        if (USE_CANNY) {
                            // https://docs.opencv.org/4.x/da/d5c/tutorial_canny_detector.html
                            thresh = new Mat(rows, cols, COLOR_BGR2GRAY);
                            Mat tmp = new Mat(rows, cols, COLOR_BGR2GRAY);
                            /* reduces the number of parts
                            Imgproc.blur(grabbedImage, tmp, new Size(3,3));*/
                            Imgproc.Canny(grabbedImage, thresh, 100, 200, 3, false);
                            /* increasing the thickness of the lines
                            float[][] maskValues = {{1, 0, 1}, {0, 1, 0}}; // (1,3) - сдвиг по горизонтали, (2,3) - по вертикали. могут быть отрицательными
                            Mat mask = new Mat(2, 3, CV_32FC1);
                            for (int i = 0; i < 2; i++)
                                for (int j = 0; j < 3; j++)
                                    mask.put(i, j, maskValues[i][j]);
                            Mat moveRight = new Mat(rows, cols, COLOR_BGR2GRAY);
                            Imgproc.warpAffine(thresh, moveRight, mask, new Size(cols, rows));
                            Core.bitwise_or(thresh, moveRight, tmp);*/
                            // invert the color of the image
                            Core.bitwise_not(thresh, tmp);
                            //thresh = tmp;
                            // it seems to be better this way?
                            blurFineLines(tmp, thresh);

                            if (OUTPUT_CANNY) {
                                BufferedImage bi = java2dFrameConverter.getBufferedImage(converter.convert(thresh));
                                ImageIO.write(bi, "png", new File(PATCH + "canny\\canny-" + vFrNumber + ".png"));
                            }
                        }
                        else if (USE_THRESH) {
                            // https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
                            thresh = new Mat(rows, cols, COLOR_BGR2GRAY);
                            if (!USE_2_THRESH) {
                                Imgproc.adaptiveThreshold(gray, thresh, 255,
                                        Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        Imgproc.THRESH_BINARY, THRESH_COEFFICIENTS.a, THRESH_COEFFICIENTS.b);
                                if (BETTER_THRESH) {
                                    for (int i = 1; i < cols - 1; i++)
                                        for (int j = 1; j < rows - 1; j++) {
                                            double c = thresh.get(j, i)[0];
                                            if (c == 255.) continue;
                                            double l = thresh.get(j, i - 1)[0];
                                            double r = thresh.get(j, i + 1)[0];
                                            double u = thresh.get(j - 1, i)[0];
                                            double d = thresh.get(j + 1, i)[0];
                                            if (l + r + u + d >= 255. * 3) {
                                                thresh.put(j, i, 255., 255., 255.);
                                            }
                                        }
                                }
                                // "lighten" the weight of the maximum black pixels
                                Mat dst = new Mat(rows, cols, thresh.type());
                                Core.add(new Mat(rows, cols, thresh.type(), new Scalar(DIFF)), thresh, dst);
                                thresh = dst;
                            } else {
                                Mat temp = new Mat(rows, cols, COLOR_BGR2GRAY);
                                Imgproc.GaussianBlur(gray, temp, new Size(5, 5), 0);
                                Imgproc.threshold(temp, thresh, 0,
                                        255,
                                        Imgproc.THRESH_BINARY + THRESH_OTSU);
                                thresh2 = new Mat(rows, cols, COLOR_BGR2GRAY);
                                Imgproc.adaptiveThreshold(thresh, thresh2, 255,
                                        Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        Imgproc.THRESH_BINARY, 3, 2);
                                Mat temp2 = thresh;
                                thresh = thresh2;
                                thresh2 = temp2;
                                // it seems to be better this way
                                blurFineLines(thresh, temp);
                                thresh = temp;
                            }
                            if (BACK_SUB) {
                                Mat invMask = new Mat();
                                Core.bitwise_not(fgMask, invMask);
                                Mat useMask = new Mat();
                                Core.bitwise_or(invMask, thresh, useMask);
                                thresh = useMask;
                            }
                            if (OUTPUT_THRESH) {
                                BufferedImage bi;
                                if (USE_2_THRESH) {
                                    bi = java2dFrameConverter.getBufferedImage(converter.convert(thresh2));
                                    ImageIO.write(bi, "png", new File(PATCH + "thresh\\thresh2-" + vFrNumber + ".png"));
                                }
                                bi = java2dFrameConverter.getBufferedImage(converter.convert(thresh));
                                ImageIO.write(bi, "png", new File(PATCH + "thresh\\thresh1-" + vFrNumber + ".png"));
                            }
                        }
                        else thresh = gray;

                        Pair<Mat, Mat> result = createUtf8Mat(thresh, gray, thresh2, vFrNumber);

                        Frame convFr = converter.convert(result.a);
                        if (HEIGHT != convFr.imageHeight) {
                            Logger.getGlobal().log(Level.INFO, "wtf");
                            int frameWidth = (int) ((double) HEIGHT / convFr.imageHeight * convFr.imageWidth);
                            convFr = java2dFrameConverter.convert(resize(java2dFrameConverter.convert(convFr), frameWidth, HEIGHT));
                        }

                        BufferedImage bi = java2dFrameConverter.getBufferedImage(convFr);
                        if (OUTPUT_FRAMES) {
                            String name = g.getFormat().matches(".*webm.*|.*mp4.*|.*m4v.*|.*mkv.*") ? "frame" : INPUT_FILE_NAME;
                            ImageIO.write(bi, "png", new File(String.format(PATCH + "frames\\%s-%03d.png", name, vFrNumber)));
                            if (SPLIT_FILL) {
                                Frame convFill = converter.convert(result.b);
                                bi = java2dFrameConverter.getBufferedImage(convFill);
                                ImageIO.write(bi, "png", new File(String.format(PATCH + "fill\\%s-%03d.png", name, vFrNumber)));
                            }
                        }

                        System.out.printf("frame-%03d%n", vFrNumber);
                        if (createdVFrNumber == 500) ProcessPixelLine.getSymbols().removeNull();
                        if (createdVFrNumber % 500 == 0) ProcessPixelLine.getSymbols().outputStatsToFile();

                        fr = convFr;

                        long currentTimeMillis = System.currentTimeMillis();
                        long leftTimeMillis = (currentTimeMillis - startTimeMillis) / createdVFrNumber
                                * ((CREATE_FRAMES > 0 ? CREATE_FRAMES : totalCreateVFrames) - createdVFrNumber);
                        System.out.println(TimeUnit.MILLISECONDS.toMinutes(leftTimeMillis) + " minutes left");
                    }

                    // if (vFrNumber <= SKIPPED_FRAMES) continue;

                    if (OUTPUT_VIDEO) {
                        fr.timestamp = timestamp;
                        recorder.setTimestamp(g.getTimestamp());
                        recorder.record(fr);
                    }

                    if (CREATE_FRAMES > 0 && CREATE_FRAMES <= createdVFrNumber) break;
                }

                System.out.println("Runtime: " + TimeUnit.MILLISECONDS.toSeconds(System.currentTimeMillis() - startTimeMillis) + "s");
                if (OUTPUT_VIDEO) {
                    recorder.stop();
                    recorder.release();
                }
            }
            g.stop();
        } catch (Exception e) {
            e.printStackTrace();
            FFmpegLogCallback.set();
        }
    }
}
//BufferedImage biMask = java2dFrameConverter2.getBufferedImage(converter.convert(fgMask));
//ImageIO.write(biMask, "png", new File(patch + "mask\\mask-frame-" + f + ".png"));

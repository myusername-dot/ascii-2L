package com.uvideo;

import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.jetbrains.annotations.NotNull;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;
import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.logging.Level;
import java.util.logging.Logger;

import static com.uvideo.CharacterSet.*;
import static com.uvideo.MainClass.*;
import static org.opencv.core.CvType.CV_8UC1;
import static org.opencv.imgproc.Imgproc.getRotationMatrix2D;
import static org.opencv.imgproc.Imgproc.warpAffine;

public class ProcessPixelLine implements ProcessLine<Mat> {

    /**A fill is a character that is added to an image if there is no character representing
     * the outline at that position and the gray image is below the FILL_DEPTH.
     * The fill characters are placed in the fillNumbersStatic array in the order sorted by
     * file name. They are of two types: FILLING and FILLING_SOLO. FILLING_SOLO are single
     * fill characters, they are distributed by the weight of the gray image pixel:
     * fillNumbers.get((int) ((fillNumbers.size()) * pixel / FILL_DEPTH)).
     * FILLING is when a color range is represented by a set of fill characters. The file name
     * looks like xxx_filling_yy xxx-an immutable value that indicates that the characters
     * belong to the same layer. yy corresponds to the order in the layer 01..02..nn. xxx does
     * not directly indicate the range, only correlates the characters with each other, the
     * order is also determined by sorting by the name of the halyard.
     * fillNumbers.add(n.clone().setIterWithN1N2(numberF, numberL)) - a function that does
     * not change the character order of the same layer in different frames.
     * fillNumbers.add(n.clone().setIterWithN1N2V2(numberF, numberL)) - a function that changes
     * the order in one layer over time.
     * */

    private static CharacterSet<Mat> symbols;
    private static List<FillRingList> fillSNumbersStatic;
    //private final int LINE_NUMBER;
    //private final int FRAME_NUMBER;
    private final List<FillRingList> fillSNumbers;
    private final Mat threshLine, grayLine, thresh2Line;
    private final Mat dstLine, fillLine;
    private final StringBuffer dstTextLine;
    private final CountDownLatch latch;

    public static int setSymbols(List<File> sImages, List<Character> chars) throws IllegalArgumentException {
        if (sImages == null || sImages.size() == 0)
            throw new IllegalArgumentException("sImages == null || sImages.size() == 0");

        List<Integer> flags = new ArrayList<>(sImages.size());
        for (File sImage : sImages) {
            String name = sImage.getName();
            if (name.contains("_false")) {
                Logger.getGlobal().log(Level.INFO, "set flag -1 " + name);
                flags.add(FALSE);
            } else if (name.contains("_dont_move_x")) {
                Logger.getGlobal().log(Level.INFO, "set flag 1 " + name);
                flags.add(DONT_MOVE_X);
            } else if (name.matches("\\d{3}_filling_\\d{2}\\D*")) {
                int number = Integer.parseInt(name.substring(0, 3));
                Logger.getGlobal().log(Level.INFO, "set flag " + (2000 + number) + " " + name);
                flags.add(FILLING + number);
            } else if (name.contains("_filling")) {
                Logger.getGlobal().log(Level.INFO, "set flag 2 " + name);
                flags.add(FILLING_SOLO);
            } else if (name.contains("_dont_move")) {
                Logger.getGlobal().log(Level.INFO, "set flag 3 " + name);
                flags.add(DONT_MOVE);
            } else if (name.contains("_dont_spin")) {
                Logger.getGlobal().log(Level.INFO, "set flag 4 " + name);
                flags.add(DONT_SPIN);
            } else flags.add(DEFAULT);
        }
        List<Mat> symbols, tempSymbols = new ArrayList<>(sImages.size());
        sImages.forEach(i -> tempSymbols.add(Imgcodecs.imread(i.getAbsolutePath(), CV_8UC1)));

        if (SPIN) {
            symbols = new ArrayList<>(tempSymbols.size() * 3);
            Java2DFrameConverter java2dFrameConverter = new Java2DFrameConverter();
            OpenCVFrameConverter.ToOrgOpenCvCoreMat converter = new OpenCVFrameConverter.ToOrgOpenCvCoreMat();

            int count = 0;
            for (Mat symbol : tempSymbols) {
                Mat white = new Mat(symbol.rows(), symbol.cols(), CV_8UC1, new Scalar(255));
                Mat invSymbol = new Mat(symbol.rows(), symbol.cols(), CV_8UC1);
                Core.subtract(white, symbol, invSymbol);
                Mat temp = rotate(invSymbol, 8.);
                Mat rLeft = new Mat(symbol.rows(), symbol.cols(), CV_8UC1);
                Core.subtract(white, temp, rLeft);
                temp = rotate(invSymbol, -8.);
                Mat rRight = new Mat(symbol.rows(), symbol.cols(), CV_8UC1);
                Core.subtract(white, temp, rRight);
//                try {
//                    BufferedImage bi = java2dFrameConverter.getBufferedImage(converter.convert(rLeft));
//                    ImageIO.write(bi, "png", new File(MainClass.PATCH + "rotated_symbols\\s-" + count + "-l.png"));
//                    bi = java2dFrameConverter.getBufferedImage(converter.convert(rRight));
//                    ImageIO.write(bi, "png", new File(MainClass.PATCH + "rotated_symbols\\s-" + count + "-r.png"));
//                } catch (IOException e) {
//                    e.printStackTrace();
//                }
                symbols.add(symbol);
                symbols.add(rLeft);
                symbols.add(rRight);
                count++;
            }
        } else symbols = tempSymbols;
//        for (Mat s : symbols) {
//            for (int i = 0; i < s.rows(); i++)
//                for (int j = 0; j < s.cols(); j++) {
//                    double v = s.get(i, j)[0];
////                    if (v <= 255. - MainClass.DIFF + 30. && v > 255. - MainClass.DIFF) { // 170..140
////                        v -= -255. + MainClass.DIFF + 1.;
//                    if (v <= 255. - DIFF + 20.) {
//                        v -= 20;
//                        if (v < 0.) v = 0.;
//                        s.put(i, j, v);
//                    }
//                }
//        }

        ProcessPixelLine.symbols = new CharacterSet<>(Mat.class, symbols, flags, chars, 0.75);

        fillSNumbersStatic = new ArrayList<>();
        List<Integer> symbolsFlags = ProcessPixelLine.symbols.getFlags();
        int currentLayerNumber = -1;
        for (int i = 0; i < symbolsFlags.size(); i++) {
            int flag = symbolsFlags.get(i);
            int sNumber = SPIN ? i * 3 : i;
            if (flag == FILLING_SOLO) fillSNumbersStatic.add(new FillRingList(
                    new Pair<>(symbols.get(sNumber).cols(), sNumber)));
            else if (flag / FILLING == 1) {
                int number = flag % FILLING;
                if (currentLayerNumber != number) {
                    fillSNumbersStatic.add(new FillRingList(new Pair<>(symbols.get(sNumber).cols(), sNumber)));
                    currentLayerNumber = number;
                } else fillSNumbersStatic.get(fillSNumbersStatic.size() - 1)
                        .add(new Pair<>(symbols.get(sNumber).cols(), sNumber));
            }
        }

        return symbols.get(0).rows();
    }

    private static Mat rotate(Mat src, double angle) {
        Mat dst = new Mat(src.rows(), src.cols(), src.type());
        Point pt = new Point(src.cols() / 2., src.rows() / 2.);
        Mat r = getRotationMatrix2D(pt, angle, 1.0);
        warpAffine(src, dst, r, new Size(src.cols(), src.rows()));
        return dst;
    }

    public static CharacterSet<Mat> getSymbols() {
        if (symbols == null)
            throw new NullPointerException("symbols are null, use setSymbols()");
        return symbols;
    }

    private ProcessPixelLine(@NotNull Mat thresh1Line, Mat grayLine, Mat thresh2Line,
                             CountDownLatch latch, int numberF, int numberL, boolean swap) {
        if (symbols == null)
            throw new NullPointerException("symbols are null, use setSymbols()");
        if (/*threshLine.type() != CV_8U || */thresh1Line.rows() != MainClass.SYMBOL_HEIGHT || thresh1Line.cols() < 100)
            throw new IllegalArgumentException("threshLine.rows() != 14 || threshLine.cols() < 100");

        //LINE_NUMBER = numberL;
        //FRAME_NUMBER = numberF;
        this.threshLine = thresh1Line;
        this.grayLine = grayLine;
        this.thresh2Line = thresh2Line;
        dstLine = new Mat(thresh1Line.rows(), thresh1Line.cols(), CV_8UC1, new Scalar(BAW ? 0 : 255));
        if (symbols.haveChars()) dstTextLine = new StringBuffer(thresh1Line.cols() / SYMBOL_HEIGHT / 2);
        else dstTextLine = null;
        fillLine = new Mat(thresh1Line.rows(), thresh1Line.cols(), CV_8UC1, new Scalar(BAW ? 0 : 255));

        fillSNumbers = new ArrayList<>(fillSNumbersStatic.size());
        for (var n : fillSNumbersStatic) {
            try {
                if (swap) fillSNumbers.add(n.clone().setIterWithFLSwap(numberF, numberL));
                else fillSNumbers.add(n.clone().setIterWithFL(numberF, numberL));
            } catch (CloneNotSupportedException e) {
                e.printStackTrace();
            }
        }
        this.latch = latch;
    }

    public ProcessPixelLine(Mat threshLine, CountDownLatch latch) {
        this(threshLine, null, null, latch, -1, -1, false);
    }

    public ProcessPixelLine(Mat threshLine, Mat grayLine, int numberF, int numberL, CountDownLatch latch) {
        this(threshLine, grayLine, null, latch, numberF, numberL, false);
    }

    public ProcessPixelLine(Mat thresh1Line, Mat grayLine, Mat thresh2Line, CountDownLatch latch) {
        this(thresh1Line, grayLine, thresh2Line, latch, -1, -1, false);
    }

    private double compare(int pos, Mat symbol, double cCr, final int flagH) {
        double sum = 0;
        for (int i = 0; i < symbol.rows(); i++)
            for (int j = 0; j < symbol.cols(); j++) {
                double s, t, subMod;
                if (flagH == 0) s = symbol.get(i, j)[0];
                else if (flagH == 1 && i != 0) s = symbol.get(i - 1, j)[0];
                else if (flagH == -1 && i != symbol.rows() - 1) s = symbol.get(i + 1, j)[0];
                else s = 255;
                t = threshLine.get(i, pos + j)[0];
                subMod = s - t;
                if (subMod < 0) subMod = -subMod;
                if (subMod <= DIFF) continue;
                sum += subMod;
            }
        return sum / (Math.pow(symbol.cols(), symbols.getCOEFFICIENT()) * (1 + (0. + cCr) * .15)) * 10;
    }

    private double width3Compare(int left, Mat symbol, double cCr, int flag) {
        // Center
        double diff, bestC = compare(left + SYMBOL_HORIZONTAL_SHIFT, symbol, cCr, 0);
        if (flag == DONT_MOVE) return bestC;
        diff = compare(left + SYMBOL_HORIZONTAL_SHIFT, symbol, cCr, 1);
        if (bestC > diff) bestC = diff;
        diff = compare(left + SYMBOL_HORIZONTAL_SHIFT, symbol, cCr, -1);
        if (bestC > diff) bestC = diff;
        //if (bestC < 50) return 0;
        if (flag != DONT_MOVE_X) {
            // Left
            diff = compare(left, symbol, cCr, 0);
            if (bestC > diff) bestC = diff;
            diff = compare(left, symbol, cCr, 1);
            if (bestC > diff) bestC = diff;
            diff = compare(left, symbol, cCr, -1);
            if (bestC > diff) bestC = diff;
            //if (bestC < 50) return 0;
            // Right
            diff = compare(left + SYMBOL_HORIZONTAL_SHIFT * 2, symbol, cCr, 0);
            if (bestC > diff) bestC = diff;
            diff = compare(left + SYMBOL_HORIZONTAL_SHIFT * 2, symbol, cCr, 1);
            if (bestC > diff) bestC = diff;
            diff = compare(left + SYMBOL_HORIZONTAL_SHIFT * 2, symbol, cCr, -1);
            if (bestC > diff) bestC = diff;
            //if (bestC < 50) return 0;
        }

        return bestC;
    }

    private int sSelect(int pos) {
        int width = threshLine.cols() - pos;
        if (width < 8) return -1;
        final int shiftN = 0;
        int best = -1;
        double bestC = Double.MAX_VALUE;

        for (int i = 0; i < symbols.size(); i++) {
            if (!symbols.isValid(i)) continue;
            Mat symbol = symbols.get(i);
            if (width - symbol.cols() <= SYMBOL_HORIZONTAL_SHIFT + 1) continue;

            if (i == shiftN) {
                double diff = compare(pos, symbol, symbols.getCorrection(i), 0);
                if (diff < 300) return i;
                bestC = diff;
                best = i;
                if (SPIN) i += 2;
                continue;
            }

            int flag = symbols.getFlag(i);
            if (SPIN && i % 3 != 0 && (flag == DONT_SPIN || flag == DONT_MOVE)) continue;

            double diff = width3Compare(pos - SYMBOL_HORIZONTAL_SHIFT, symbol, symbols.getCorrection(i), flag);
            if (diff == 0) return i;
            if (bestC > diff) {
                bestC = diff;
                best = i;
            }
        }

        return best;
    }

    private void addPixSymbol(Mat symbol, int pos, boolean isFilling) {
        if (BAW) {
            Mat temp = new Mat(symbol.rows(), symbol.cols(), CV_8UC1);
            Core.subtract(new Mat(symbol.rows(), symbol.cols(), CV_8UC1, new Scalar(255.)), symbol, temp);
            symbol = temp;
        }
        if (SPLIT_FILL && isFilling)
            symbol.copyTo(fillLine.submat(new Rect(pos, 0, symbol.cols(), symbol.rows())));
        else
            symbol.copyTo(dstLine.submat(new Rect(pos, 0, symbol.cols(), symbol.rows())));
    }

    @Override
    public void run() {
        // sorry for this bad code, it's quite compact and there's a lot going on here

        //final Random random = new Random(FRAME_NUMBER + LINE_NUMBER);
        int widthPix = threshLine.cols();
        int posPix = 5, leftPix = widthPix - 5, shiftSize = symbols.get(0).cols();
        final int shiftNumber = 0;

        // waitNextSpace - if the fill symbol is wide enough relative to the space, we can't put it right away,
        // as it may get on the future main contour symbol. instead, we go through two circles of the loop, if
        // on the second circle again a space falls out of the main contour, and the background again meets the
        // fill requirement (pixel < FILL_DEPTH), we move back and put a wide character.
        boolean isFillChar, waitNextSpace = false;
        while (leftPix > 10) {
            isFillChar = false;
            int sNumber = sSelect(posPix);
            if (sNumber == -1) break;

            Mat symbol = symbols.getWithInc(sNumber);

            if (grayLine != null && sNumber == shiftNumber) {
                // checking the gray pixel behind the symbol
                double pixel = grayLine.get(grayLine.rows() / 2, posPix + symbol.cols() / 2)[0];
                // if there is a second thresh, make sure that there is no dark pixel on it
                if (pixel < FILL_DEPTH && fillSNumbers.size() > 0 && (thresh2Line == null ||
                        thresh2Line.get(thresh2Line.rows() / 2, posPix + symbol.cols() / 2)[0] < 200.)) {
                    // move to the previous position and put a wide symbol
                    if (waitNextSpace) posPix -= shiftSize;
                    // we select the fill according to the brightness of the pixel
                    Pair<Integer, Integer> shiftPAndSNumber = fillSNumbers
                            .get((int) ((fillSNumbers.size()) * pixel / FILL_DEPTH)).next(posPix);
                    sNumber = shiftPAndSNumber.b;
                    symbol = symbols.get(sNumber);

                    if (!waitNextSpace && symbol.cols() > Math.ceil(shiftSize * 1.5)) {
                        // we will use the wide character in the next step, if the main character is again a space
                        waitNextSpace = true;
                        // in the meantime, we assign the character as a space
                        sNumber = shiftNumber;
                        symbol = symbols.get(sNumber);
                    } else // the alignment is applied when we put the fill symbol
                        posPix += shiftPAndSNumber.a; // if FILL_ALIGNMENT is disabled, this value is always 0

                    if (widthPix - posPix - symbol.cols() < 2) break;
                    isFillChar = true;
                } else waitNextSpace = false;
            } else waitNextSpace = false;

            // writing a text character
            Optional<Character> charOp = symbols.getChar(sNumber);
            if (dstTextLine != null && charOp.isPresent()) {
                char c = charOp.get();
                //if (c == 'r') c = (char) (random.nextInt(8) + 50);
                // it will work in the second round, just before waitNextSpace becomes false
                if (sNumber != shiftNumber && waitNextSpace)
                    dstTextLine.deleteCharAt(dstTextLine.length() - 1); // remove the extra space
                dstTextLine.append(c);
            }

            // printing a pixel character
            // the pixel line is filled with white or black pixels by default
            if (sNumber != shiftNumber) {
                addPixSymbol(symbol, posPix, isFillChar);
                posPix += SYMBOL_SPACING;
                waitNextSpace = false;
            }

            posPix += symbol.cols();
            leftPix = widthPix - posPix;
        }

        latch.countDown();
    }

    public Mat getResult() {
        return dstLine;
    }

    public Mat getFill() {
        return fillLine;
    }

    public String getTextResult() {
        if (dstTextLine != null) return dstTextLine.toString();
        else return "";
    }

    public List<FillRingList> getFillSNumbers() {
        return fillSNumbers;
    }
}

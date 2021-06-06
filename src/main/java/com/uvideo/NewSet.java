package com.uvideo;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_java;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.LinkedList;

import static org.opencv.core.Core.NORM_MINMAX;
import static org.opencv.core.CvType.CV_8UC1;
import static org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY;

public class NewSet {

    /**
     * _default - these characters have the greatest advantage in matching.
     * _dont_move_x - symbols that are useless or harmful to move on the x-axis. For example,
     * because of the movement on the x, the character may be displayed where it should not be
     * _dont_spin - symbols that you want to see more likely, but do not want to spend a lot of
     * time processing, or symbols that are useless to rotate.
     * _dontMove - rare characters and the ones you want to spend the least time on. This may
     * include punctuation marks, since if you use movements and turns for them, they will appear
     * where they are not needed.
     * _filling - symbols that do not participate in the main selection. The threshold value is
     * changed in the MainClass of the FILL_DEPTH variable. An element can be represented by
     * a character, a word, or a string. The length of each element does not affect the
     * distribution of the gradient, which is distributed evenly: FILL_DEPTH / _filling.length.
     * _false - this will not be created. But you can add the string _false to the name of an
     * already created file if it appears too often or you don't want it to be processed and
     * displayed. The main thing is that the change does not affect the sorting order of the
     * directory by file names, if you add a string to the end of the name, this will not happen.
     */

    private static String fontPatch = "data_set\\fonts\\MS Gothic.ttf";
    private static final int size = 14;
    // the first character must always be a space
    private static final String spaceAnd_default = " /／＼|｜ﾉ＞l∨∧＜";
    private static final String _dontMoveX = "-ｰ一ﾆ二ニ＝ヽﾍ丶ゝ=";
    private static final String _dontSpin = "へｲイ十トΤ>ﾝＯ7Yﾚ^７く<七LvVレｿ∠γメ┴フ(ソう";
    private static final String _dontMove = "¯_\"\\×〃,.{}';乂ｌｉ!iィﾊ从ｨﾔfcハﾞ┤弋シﾄ〝〟`ｧｒア∈三Ｏrｊjいヾ[八1ァﾑ人∠ゝﾟム" +
            "ミ斗孑≠ﾘん心》こツ≧厶公ｱｪxｭ少t彡込≫チoX厂芹」「└┘┐小リ〔ﾏ廴示ゞ云NしZz()├〈〉";
    private static final String[] _filling = {"ﾐ", "*", "+", ":"};
    private static final String _false = "￣＿｀―弐ー．′‐─、¨´‘’゛（）⌒";

    private static void createCharsTxt(LinkedList<Pair<String, String>> symbols, String outPatch) {
        int count = 0;
        File chars = new File(outPatch + "chars.txt");
        try (PrintWriter writer = new PrintWriter(chars, StandardCharsets.UTF_8)) {
            for (var e : symbols) {
                count++;
                for (Character c : e.b.toCharArray())
                    writer.println(String.format("%03d_%s", count, c));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static float chooseSize(int pixDst) throws IOException, FontFormatException {
        BufferedImage img = new BufferedImage(1, 1, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = img.createGraphics();
        float fontSize = pixDst;
        int pixSize = pixDst + 1, counter = 0;
        while (pixSize != pixDst) {
            if (pixSize < pixDst) fontSize += .5f;
            if (pixSize > pixDst) fontSize -= .5f;
            Font font = Font.createFont(Font.TRUETYPE_FONT,
                    new File(fontPatch)).deriveFont(fontSize);
            g2d.setFont(font);
            FontMetrics fm = g2d.getFontMetrics();
            pixSize = fm.getHeight();
            counter++;
            if (counter == 5) throw new FontFormatException("can't get a font of this size");
        }
        return fontSize;
    }

    private static LinkedList<Pair<String, String>> buildSTrain() {
        LinkedList<Pair<String, String>> train = new LinkedList<>();
        // adding basic symbols and flags for them
        for (Character c : spaceAnd_default.toCharArray())
            train.add(new Pair<>("", c.toString()));
        for (Character c : _dontMoveX.toCharArray())
            train.add(new Pair<>("_dont_move_x", c.toString()));
        for (Character c : _dontSpin.toCharArray())
            train.add(new Pair<>("_dont_spin", c.toString()));
        for (Character c : _dontMove.toCharArray())
            train.add(new Pair<>("_dont_move", c.toString()));
        // adding a gradient
        for (String f : _filling)
            train.add(new Pair<>("_filling", f));
        return train;
    }

    public static void main(String[] args) throws Exception {
        if (args.length > 0) fontPatch = args[0];
        //it is best to use 14-16
        int fontPixSize = args.length > 1 ? Integer.parseInt(args[1]) : size;
        float fontSize = chooseSize(fontPixSize);

        System.out.println("Start loading OpenCV Java native library...");
        Loader.load(opencv_java.class);
        System.out.println("Loading done");

        String outPatch = "data_set\\" + fontPatch.substring(fontPatch.lastIndexOf("\\") + 1)
                .replace(' ', '_') + "_" + fontPixSize + "_00\\";
        File folder = new File(outPatch);
        int count = 0;
        // looking for an empty folder
        while (folder.exists()) {
            outPatch = String.format("%s%02d%s", outPatch.substring(0, outPatch.length() - 3), count, "\\");
            folder = new File(outPatch);
            count++;
        }
        folder.mkdir();

        var train = buildSTrain();

        createCharsTxt(train, outPatch);

        count = 0;
        for (var pair : train) {
            count++;
            int size = pair.b.toCharArray().length, countFill = 0;
            // a fill can contain multiple characters
            for (Character c : pair.b.toCharArray()) {
                countFill++;
                BufferedImage img = new BufferedImage(1, 1, BufferedImage.TYPE_INT_RGB);
                Graphics2D g2d = img.createGraphics();
                Font font = Font.createFont(Font.TRUETYPE_FONT,
                        new File(fontPatch)).deriveFont(fontSize);
                // new Font("Arial", Font.PLAIN, 15);
                g2d.setFont(font);
                FontMetrics fm = g2d.getFontMetrics();
                int width = fm.charWidth(c);
                int height = fm.getHeight();
                g2d.dispose();

                img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
                g2d = img.createGraphics();
                g2d.setColor(Color.WHITE);
                g2d.fillRect(0, 0, width, height);
                g2d.setRenderingHint(RenderingHints.KEY_ALPHA_INTERPOLATION,
                        RenderingHints.VALUE_ALPHA_INTERPOLATION_QUALITY);
                g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                        RenderingHints.VALUE_ANTIALIAS_ON);
                g2d.setRenderingHint(RenderingHints.KEY_COLOR_RENDERING,
                        RenderingHints.VALUE_COLOR_RENDER_QUALITY);
                g2d.setRenderingHint(RenderingHints.KEY_DITHERING,
                        RenderingHints.VALUE_DITHER_ENABLE);
                g2d.setRenderingHint(RenderingHints.KEY_FRACTIONALMETRICS,
                        RenderingHints.VALUE_FRACTIONALMETRICS_ON);
                g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                        RenderingHints.VALUE_INTERPOLATION_BILINEAR);
                g2d.setRenderingHint(RenderingHints.KEY_RENDERING,
                        RenderingHints.VALUE_RENDER_QUALITY);
                g2d.setRenderingHint(RenderingHints.KEY_STROKE_CONTROL,
                        RenderingHints.VALUE_STROKE_PURE);
                g2d.setFont(font);
                fm = g2d.getFontMetrics();
                g2d.setColor(Color.BLACK);
                g2d.drawString(String.valueOf(c), 0, fm.getAscent());
                g2d.dispose();

                if (c != ' ') {
                    // skip the space bar and normalize the images
                    OpenCVFrameConverter.ToOrgOpenCvCoreMat converter = new OpenCVFrameConverter.ToOrgOpenCvCoreMat();
                    Java2DFrameConverter java2dFrameConverter = new Java2DFrameConverter();
                    Mat matImg = converter.convert(java2dFrameConverter.getFrame(img));
                    Mat gray = new Mat(matImg.rows(), matImg.cols(), COLOR_BGR2GRAY);
                    Imgproc.cvtColor(matImg, gray, COLOR_BGR2GRAY);
                    Mat normalizeImg = new Mat(height, width, CV_8UC1);
                    Core.normalize(gray, normalizeImg, 0, 255, NORM_MINMAX, CV_8UC1);
                    img = java2dFrameConverter.getBufferedImage(converter.convert(normalizeImg));
                }
                if (size == 1)
                    ImageIO.write(img, "png", new File(String.
                            format(outPatch + "%03d%s.png", count, pair.a)));
                else
                    ImageIO.write(img, "png", new File(String.
                            format(outPatch + "%03d%s_%02d.png", count, pair.a, countFill)));
            }
        }
        System.out.println("symbols created, folder: " + outPatch.substring(outPatch
                .indexOf('\\') + 1, outPatch.lastIndexOf('\\')));
    }
}

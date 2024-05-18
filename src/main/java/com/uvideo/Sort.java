package com.uvideo;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.opencv_java;
import org.opencv.core.Mat;

import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Collectors;

public class Sort {
    private static final String fontPatch = "data_set\\fonts\\IPAGothic Regular.ttf";
    private static final int fontPixSize = 14;
    private static final String values = "ﾐABCDEFGHIJKLMNOPQRSTUVWXYZ&zyxwvutsrqponmlkjihgfedcba1234567890@#\"«*=+-;:.,⋅";

    public static void main(String[] args) throws IOException, FontFormatException {
        Font mainFont = Font.createFont(Font.TRUETYPE_FONT, new File(fontPatch)).deriveFont((float) fontPixSize);
        mainFont = NewSet.chooseSize(mainFont, fontPixSize);
        Font fallbackFont = new Font("Serif", Font.PLAIN, fontPixSize);
        fallbackFont = NewSet.chooseSize(fallbackFont, fontPixSize);

        System.out.println("Start loading OpenCV Java native library...");
        Loader.load(opencv_java.class);
        System.out.println("Loading done");

        int[] codePoints = values.codePoints().toArray();
        List<Pair<Integer, Double>> arithmeticMeanWeights = new LinkedList<>();
        for (int n = 0; n < codePoints.length; n++) {
            int codePoint = codePoints[n];
            Mat charMat = NewSet.createNormalizedCharMat(mainFont, fallbackFont, codePoint);
            double sum = 0.;
            for (int i = 0; i < charMat.rows(); i++)
                for (int j = 0; j < charMat.cols(); j++) {
                    sum += 255. - charMat.get(i, j)[0];
                }
            sum /= charMat.rows() * charMat.cols();
            arithmeticMeanWeights.add(new Pair<>(codePoint, sum));
        }

        arithmeticMeanWeights = arithmeticMeanWeights.stream()
                .sorted(Comparator.comparing(e -> e.b, Comparator.reverseOrder()))
                .collect(Collectors.toList());

        arithmeticMeanWeights.forEach(e -> System.out.print("\"" + new String(Character.toChars(e.a)) + "\", "));
        System.out.println();

        String unnecessary = null;
        do {
            if (unnecessary != null) {
                List<Integer> unnecessaryCp = unnecessary.codePoints().boxed().toList();
                arithmeticMeanWeights.removeIf(e -> unnecessaryCp.contains(e.a));
            }
            System.out.println("Recommended:");
            int targetSize = 20;
            double maxWeight = arithmeticMeanWeights.getFirst().b;
            double minWeight = arithmeticMeanWeights.getLast().b;
            double interval = maxWeight - minWeight;
            double step = interval / targetSize;
            int i = 0;
            System.out.println(minWeight);
            System.out.println(maxWeight);
            for (double w = maxWeight; w > minWeight; w -= step) {
            /*System.out.println(arithmeticMeanWeights.get(i + 1).b);
            System.out.println(arithmeticMeanWeights.get(i).b);
            System.out.println("i " + i + " " + "w " + w + " " + Math.abs(arithmeticMeanWeights.get(i + 1).b - w) + " < " + Math.abs(arithmeticMeanWeights.get(i).b - w));*/
                while (arithmeticMeanWeights.size() > i + 1
                        && Math.abs(arithmeticMeanWeights.get(i + 1).b - w) <= Math.abs(arithmeticMeanWeights.get(i).b - w)) {
                    i++;
                }
                System.out.print("\"" + new String(Character.toChars(arithmeticMeanWeights.get(i).a)) + "\", ");
            }
            System.out.println();
            System.out.print("Enter the characters you want to change or \"end\" command: ");
            Scanner in = new Scanner(System.in);
            unnecessary = in.next();
        } while (!"end".equals(unnecessary));
    }
}

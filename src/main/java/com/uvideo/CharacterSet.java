package com.uvideo;

import com.google.common.primitives.Booleans;
import org.opencv.core.Mat;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.Array;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Predicate;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static com.uvideo.MainClass.*;

public class CharacterSet<T> {

    /**I used to think about using vector images, and this class was supposed to be universal, but
     * I didn't find an effective way to compare vector images other than converting them to pixels.
     *
     * symbols - (size) the list of all characters, with the SPIN option, includes all rotated characters,
     *  even those with the DONT_SPIN and DONT_MOVE flags in order: normal character, rotated left,
     *  rotated right; next character. the content does not change in any way.
     * size - number of characters.
     * uniqueSize - number of unique, non-rotated characters.
     * used - (uniqueSize) the number of times the symbol was applied. changes during parallel processing.
     * valid - (uniqueSize) valid characters that make up currentUSize. change to " remove.+" functions in
     *  single-threaded mode.
     * currentUSize - the number of valid characters (not marked with a FALSE flag or rejected during
     *  the call of the corresponding functions) without filling characters.
     * correction(cCr) - (size) compensates for the mismatch of a larger number of pixels in heavy characters.
     * chars - (size) list of characters from the file chars.txt, which may be missing(to support older
     *  character sets). does not change after loading.
     * flags - (uniqueSize) flags obtained from file names.
     * */
    private final T[] symbols;
    private final int size;
    private final int uniqueSize;
    private int currentUSize;
    private final AtomicLong[] used;
    private final boolean[] valid;
    private final double[] correction;
    private final int[] flags;
    private final char[] chars;
    // ~~~~~ Flags ~~~~~
    public static final int DEFAULT = 0;
    public static final int FALSE = -1;
    public static final int DONT_MOVE_X = 1;
    public static final int FILLING = 2000;
    public static final int FILLING_SOLO = 2;
    public static final int DONT_MOVE = 3;
    public static final int DONT_SPIN = 4;

    public CharacterSet(Class<T> clazz, List<T> symbols, List<Integer> flags, List<Character> chars) {
        if (symbols == null) throw new NullPointerException("symbols == null");
        size = symbols.size();
        if (chars != null && !chars.isEmpty()) {
            if (chars.size() != flags.size())
                throw new IllegalArgumentException("chars != null && chars.size() != 0 && chars.size() != flags.size()");
            if (SPIN && size / 3 != chars.size() || !SPIN && size != chars.size())
                throw new IllegalArgumentException("chars != null && chars.size() != 0 && SPIN && symbols.size() / 3 != chars.size() || !SPIN && symbols.size() != chars.size()");
        }
        if (SPIN && size / 3 != flags.size() || !SPIN && size != flags.size())
            throw new IllegalArgumentException("SPIN && symbols.size() / 3 != flags.size() || !SPIN && symbols.size() != flags.size()");
        if (SPIN) {
            if (size % 3 != 0) throw new IllegalArgumentException("symbols.size() % 3 != 0 && MainClass.SPIN");
            uniqueSize = size / 3;
        } else uniqueSize = size;

        this.symbols = (T[]) Array.newInstance(clazz, size);
        this.flags = new int[uniqueSize];
        if (chars != null && !chars.isEmpty()) this.chars = new char[uniqueSize];
        else this.chars = null;
        used = new AtomicLong[uniqueSize];
        valid = new boolean[uniqueSize];
        correction = new double[size];

        for (int i = 0; i < uniqueSize; i++) {
            int flag = flags.get(i);
            this.flags[i] = flag;
            used[i] = new AtomicLong(0L);
            valid[i] = flag != FILLING_SOLO && flag != FALSE && flag / FILLING != 1;
            if (this.chars != null) this.chars[i] = chars.get(i);
        }

        for (int i = 0; i < size; i++) {
            T symbol = symbols.get(i);
            this.symbols[i] = symbol;
            double cCr = 1;
            if (symbol instanceof Mat) {
                if (i != 0) { // skip space
                    Mat s = (Mat) symbol;
                    double halfRows = s.rows() / 2.;
                    double sum = 0;
                    for (int c = 0; c < s.cols(); c++)
                        for (int r = 0; r < s.rows(); r++)
                            sum += 255. - s.get(r, c)[0];

                    //ScCr = 1.2
                    //cCr = Math.pow(sum, 1.2) / s.rows() / s.cols() / 500. + (double) s.cols() / halfRows; // ++
                    //cCr = Math.pow(sum, 1.16) / s.rows() / s.cols() / 255. + (double) s.cols() / halfRows * 0.7;
                    //cCr = Math.pow(sum, 1.5) / s.rows() / s.cols() / 10000. * (1. + (double) s.cols() / halfRows * 0.2) + (double) s.cols() / halfRows * 0.7;
                    //ScCr = 1.1;1?
                    //cCr = Math.pow(sum, 1.2) / s.rows() / s.cols() / 500. * (1. + (double) s.cols() / halfRows * 0.2) + (double) s.cols() / halfRows * 0.7; // +
                    //cCr = Math.pow(sum, 1.15) / s.rows() / Math.pow(s.cols(), 0.75) / 255. + Math.pow(s.cols(), 0.75) / halfRows;
                    //ScCr = 1
                    //cCr = (sum / s.rows() / Math.pow(s.cols(), 0.5) / 255. + (double) s.cols() / halfRows * 0.5) * 0.7 + 0.5; // !!++
                    //cCr = (Math.pow(0.5 + sum / s.rows() / Math.pow(s.cols(), 0.5) / 255., 2) + (double) s.cols() / halfRows * 0.6) * 0.7 + 0.5;
                    cCr = 0.5 + sum / s.rows() / Math.pow(s.cols(), 0.75) / 125.;
                    //cCr = (Math.pow(sum / s.rows(), 1.35) / 10000. + /*sum / s.rows() / Math.pow(s.cols(), 0.75) / 125. +*/ (double) s.cols() / halfRows * 0.5) * 0.7 + 0.5; //+
                    //cCr = (Math.pow(sum / s.rows(), 1.75) / 200000. + sum / s.rows() / Math.pow(s.cols(), 0.5) / 255. + (double) s.cols() / halfRows * 0.6) * 0.7 + 0.5; //+?
                    //cCr = Math.pow(sum / s.rows() / s.cols() / 70, 2) + (double) s.cols() / halfRows; // ++
                    //cCr = 1. + sum / 255. / s.rows() / Math.pow(s.cols(), 0.75);
                    //ScCr = 0.8
                    //cCr = Math.pow(sum, 1.4) / s.rows() / s.cols() / 4500. * (1. + (double) s.cols() / halfRows * 0.2) + (double) s.cols() / halfRows * 0.6; // +
                    //ScCr = 1.3
                }
                if (SPIN && i % 3 == 0) System.out.println("cCr " + i + "= " + cCr);
            } else {
                Logger.getGlobal().log(Level.WARNING, "!symbol instanceof Mat");
                break;
            }
            correction[i] = cCr;
        }

        currentUSize = (int) (Booleans.asList(valid).stream().filter(Boolean::booleanValue).count());
        Logger.getGlobal().log(Level.INFO, "number of valid characters without fill: " + currentUSize);
    }

    public T get(int index) {
        return symbols[index];
    }

    public T getUnique(int index) {
        if (SPIN) index /= 3;
        return symbols[index];
    }

    public double getCorrection(int index) {
        return correction[index];
    }

    public T getWithInc(int index) {
        T symbol;
        if (SPIN) {
            used[index / 3].incrementAndGet();
            symbol = symbols[index - index % 3];
        } else {
            used[index].incrementAndGet();
            symbol = symbols[index];
        }
        return symbol;
    }

    public boolean isValid(int index) {
        if (SPIN) index /= 3;
        return valid[index];
    }

    public int getFlag(int index) {
        if (SPIN) index /= 3;
        return flags[index];
    }

    public boolean haveChars() {
        return chars != null;
    }

    public Optional<Character> getChar(int index) {
        if (this.chars == null)
            return Optional.empty();
        if (SPIN) index /= 3;
        return Optional.of(chars[index]);
    }

    public List<Integer> getNumbersWithFlag(int flag){
        List<Integer> n = new ArrayList<>();
        for (int i = 0; i < uniqueSize; i++)
            if (flags[i] == flag) n.add(i);
        return n;
    }

    public List<Integer> getNumbersWithFlags(Predicate<Integer> p){
        List<Integer> n = new ArrayList<>();
        for (int i = 0; i < uniqueSize; i++)
            if (p.test(flags[i])) n.add(i);
        return n;
    }

    public List<Integer> getFlags() {
        return Arrays.stream(flags)
                .boxed()
                .collect(Collectors.toList());
    }

    public Optional<T> getValid(int index) {
        T symbol = null;
        if (isValid(index)) symbol = symbols[index];
        return Optional.ofNullable(symbol);
    }

    public void removeMostRarelyUsed(double percent) {
        IntStream.range(1, uniqueSize)
                .boxed()
                .filter(i->valid[i])
                .collect(Collectors.toMap(i->i, i->used[i].get()))
                .entrySet()
                .stream()
                .sorted(Map.Entry.<Integer, Long>comparingByValue().reversed())
                .limit((int) (currentUSize * percent / 100.))
                .forEach(u -> valid[u.getKey()] = false);
        int before = currentUSize;
        currentUSize = (int) (Booleans.asList(valid).stream().filter(Boolean::booleanValue).count());
        Logger.getGlobal().log(Level.INFO, "Ignore " + (before - currentUSize) + " symbols");
    }

    public void removeMostOftenUsed(double percent) {
        IntStream.range(1, uniqueSize)
                .boxed()
                .filter(i->valid[i])
                .collect(Collectors.toMap(i->i, i->used[i].get()))
                .entrySet()
                .stream()
                .sorted(Map.Entry.<Integer, Long>comparingByValue())
                .limit((int) (currentUSize * percent / 100.))
                .forEach(u -> valid[u.getKey()] = false);
        int before = currentUSize;
        currentUSize = (int) (Booleans.asList(valid).stream().filter(Boolean::booleanValue).count());
        Logger.getGlobal().log(Level.INFO, "Ignore " + (before - currentUSize) + " symbols");
    }

    public void removeNull() {
        for (int i = 1; i < uniqueSize; i++)
            if (used[i].get() <= 10L) valid[i] = false;
        int before = currentUSize;
        currentUSize = (int) (Booleans.asList(valid).stream().filter(Boolean::booleanValue).count());
        Logger.getGlobal().log(Level.INFO, "Ignore " + (before - currentUSize) + " symbols");
    }

    public void outputStatsToFile() {
        File file = new File(String.format("%s%s\\stats_%s_%s.txt"
                , MainClass.PATCH, SYMBOLS_FOLDER,
                SYMBOLS_FOLDER, MainClass.getInputFileName()));
        try (PrintWriter writer = new PrintWriter(file, StandardCharsets.UTF_8)) {
            writer.println(this);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public int size() {
        return size;
    }

    @Override
    public String toString() {
        StringBuffer buffer = new StringBuffer();
        Formatter fmt = new Formatter(buffer);
        Character c = null;
        if (SPIN)
            for (int i = 0; i < size / 3; i++) {
                if (chars != null) c = chars[i];
                fmt.format("char=%s number=%d used=%d valid=%b cCr=%f flag=%d\r\n",
                        c, i + 1, used[i].get(), valid[i], correction[i * 3], flags[i]);
            }
        else
            for (int i = 0; i < size; i++) {
                if (chars != null) c = chars[i];
                fmt.format("char=%s number=%d used=%d valid=%b cCr=%f flag=%d\r\n",
                        c, i + 1, used[i].get(), valid[i], correction[i], flags[i]);
            }
        return buffer.toString();
    }
}

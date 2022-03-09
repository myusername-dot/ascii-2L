package com.uvideo;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import static com.uvideo.MainClass.FILL_ALIGNMENT;
import static com.uvideo.ProcessPixelLine.FILL_SPACING;

public class FillRingList implements Cloneable {

    /**size - size of WNLayer.
     * WNLayer - contains the value a-the width of the character, b-the number
     *  of the character in the usual order. Added in the ProcessPixelLine
     *  constructor if the character is marked as a fill character. It usually
     *  contains one element, but it can contain several if there are several
     *  characters that have FILLING flags and belong to the same xxx_fill
     *  layer.
     * pixLength - length of the entire chain with FILL_SPACING.
     * select - contains a pointer to which position corresponds to which character.
     * startPos, startPosPix - the symbol and the position at which this fill layer
     *  should start.
     * */
    private static final int fillSpacing = FILL_ALIGNMENT ? FILL_SPACING : 0;

    private List<Pair<Integer, Integer>> WNLayer;
    private Iterator<Pair<Integer, Integer>> iter;
    private int size, pixLength;
    private ArrayList<Integer> select;
    private int startPos, startPosPix;

    private void initSelect() {
        select = new ArrayList<>();
        int el = -1, pos = 0;
        for(int i = 0; i < pixLength; i++){
            if (pos <= i){
                el++;
                pos += WNLayer.get(el).a + fillSpacing;
            }
            select.add(el);
        }
    }

    FillRingList(FillRingList other){
        this.WNLayer = new ArrayList<>();
        WNLayer.addAll(other.WNLayer);
        iter = WNLayer.iterator();
        size = other.size;
        pixLength = other.pixLength;
        select = other.select;
        startPos = other.startPos;
        startPosPix = other.startPosPix;
    }

    FillRingList(List<Pair<Integer, Integer>> arr) {
        this.WNLayer = arr;
        iter = arr.iterator();
        if (!iter.hasNext())
            throw new IllegalArgumentException("the array contains no elements");
        startPos = 0;
        startPosPix = arr.get(0).a;
        size = arr.size();
        pixLength = 0;
        while (iter.hasNext())
            pixLength += iter.next().a + fillSpacing;
        iter = arr.iterator();
        initSelect();
    }

    FillRingList(Pair<Integer, Integer> p) {
        this.WNLayer = new ArrayList<>();
        WNLayer.add(p);
        iter = WNLayer.iterator();
        if (!iter.hasNext())
            throw new IllegalArgumentException("the array contains no elements");
        startPos = 0;
        startPosPix = WNLayer.get(0).a;
        size = 1;
        pixLength = p.a + fillSpacing;
        iter = WNLayer.iterator();
        initSelect();
    }

    public void add(Pair<Integer, Integer> var){
        WNLayer.add(var);
        size++;
        pixLength += var.a + fillSpacing;
        for (int i = 0; i < var.a + fillSpacing; i++)
            select.add(size - 1);
    }

    public void throwIter(){
        iter = WNLayer.iterator();
    }

    private void shiftArrLeftRound(int count, final int ignore) {
        count %= size - ignore;
        List<Pair<Integer, Integer>> newArr = new ArrayList<>(size);
        for (int i = 0; i < size - ignore; i++){
            int second = i + count;
            if (second > size - 1 - ignore) second -= size - ignore;
            newArr.add(WNLayer.get(second));
        }
        for (int i = size - ignore; i < size; i++)
            newArr.add(WNLayer.get(i));
        WNLayer = newArr;
    }

    private void shiftFirstCharacter(int count) {
        List<Pair<Integer, Integer>> newArr = new ArrayList<>(size);
        Pair<Integer, Integer> temp = WNLayer.get(0);
        for (int i = 0; i < count; i++)
            newArr.add(WNLayer.get(i + 1));
        newArr.add(temp);
        for (int i = count + 1; i < size; i++)
            newArr.add(WNLayer.get(i));
        WNLayer = newArr;
    }

    public FillRingList setIterWithFL(int frameNumber, int lineNumber){
        // moving lines
        frameNumber /= 22; // changes every n frame
        int pos = lineNumber % size;
        pos += (lineNumber % 2 == 0) ? frameNumber % size : -(frameNumber % size);
        if (pos < 0) pos += size;
        startPos = pos % size;
        startPosPix = 0;
        for (int i = 0; i < startPos; i++) {
            next();
            startPosPix += WNLayer.get(i).a + fillSpacing;
        }
        return this;
    }

    public FillRingList setIterWithFLSwap(int frameNumber, int lineLumber){
        // moving symbols
        if (size < 2) return this;
        // ignore the space character at the end
        int ignore = 1;
        // the two functions represent different epochs of shifts. shiftArrLeftRound-shift
        // the entire array to the left. shiftFirstCharacter - when a single character
        // travels to the end of the array to the right.
        shiftArrLeftRound(frameNumber / 10 / (size - ignore), ignore);
        shiftFirstCharacter(frameNumber / 10 % (size - ignore));
        initSelect();
        // after a full update, the row is shifted to the right
        frameNumber /= (size - ignore) * 10;
        int pos = lineLumber % size;
        pos += (lineLumber % 2 == 0) ? frameNumber % size : -(frameNumber % size);
        if (pos < 0) pos += size;
        startPos = pos % size;
        startPosPix = 0;
        for (int i = 0; i < startPos; i++) {
            next();
            startPosPix += WNLayer.get(i).a + fillSpacing;
        }
        return this;
    }

    public Pair<Integer, Integer> next() {
        if (!iter.hasNext())
            iter = WNLayer.iterator();
        return iter.next();
    }

    public Pair<Integer, Integer> next(int pos) {
        // the character is selected according to the position, as if the entire
        // line before it represented this fill layer.
        if (!FILL_ALIGNMENT && size == 1) return new Pair<>(fillSpacing, WNLayer.get(0).b);
        pos += startPosPix;
        int pixPos = pos % pixLength;
        int s = select.get(pixPos);
        if (!FILL_ALIGNMENT) return new Pair<>(0, WNLayer.get(s).b);
        int plus = 0;
        if (pixPos > 0 && select.get(pixPos - 1) == s) {
            do plus++;
            while (pixPos + plus < pixLength && select.get(pixPos + plus) == s);
            if (pixPos + plus == pixLength) s = 0;
            else s++;
        }
        return new Pair<>(plus, WNLayer.get(s).b);
    }

    public List<Pair<Integer, Integer>> getArr() {
        return WNLayer;
    }

    public Iterator<Pair<Integer, Integer>> getIter() {
        return iter;
    }

    public int getSize() {
        return size;
    }

    public int getPixLength() {
        return pixLength;
    }

    public ArrayList<Integer> getSelect() {
        return select;
    }

    public int getStartPos() {
        return startPos;
    }

    public int getStartPosPix() {
        return startPosPix;
    }

    @Override
    public FillRingList clone() throws CloneNotSupportedException {
        FillRingList list = (FillRingList) super.clone();
        list.throwIter();
        return list;
    }
}
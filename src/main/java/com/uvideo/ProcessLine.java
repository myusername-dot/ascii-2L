package com.uvideo;

import org.opencv.core.Mat;

import java.util.Optional;

public interface ProcessLine<T> extends Runnable {

    public T getResult();

    public T getFill();

    public String getTextResult();
}

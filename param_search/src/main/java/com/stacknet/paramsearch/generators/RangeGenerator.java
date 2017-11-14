package com.stacknet.paramsearch.generators;


public abstract class RangeGenerator<T> implements Generator<T>{
    T start;
    T end;

    public RangeGenerator(T start, T end) {
        this.start = start;
        this.end = end;
        Double d;
    }
}

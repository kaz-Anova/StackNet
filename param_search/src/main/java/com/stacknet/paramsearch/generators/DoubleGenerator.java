package com.stacknet.paramsearch.generators;

import org.apache.commons.math3.distribution.UniformRealDistribution;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;

public class DoubleGenerator extends RangeGenerator<Double> {
    private BiFunction<DoubleGenerator, Integer, List<Double>> function;
    private DoubleGenerator(Double start,
                           Double end,
                           BiFunction<DoubleGenerator, Integer, List<Double>> function) {
        super(start, end);
        this.function = function;
    }

    @Override
    public List<Double> expend(int size) {
        return function.apply(this, size);
    }

    public static DoubleGenerator getGridDoubleGenerator(Double start, Double end) {
        return new DoubleGenerator(start, end, DoubleGenerator::gridGen);
    }

    public static DoubleGenerator getUniformDoubleGenerator(Double start, Double end) {
        return new DoubleGenerator(start, end, DoubleGenerator::genUniform);
    }

    private static List<Double> genUniform(DoubleGenerator dg, int size) {
        UniformRealDistribution ud = new UniformRealDistribution(dg.start, dg.end);
        List<Double> output = new ArrayList<>(size);
        for (int i = 0; i < size; ++i) {
            output.add(ud.sample());
        }
        return output;
    }

    private static List<Double> gridGen(DoubleGenerator dg, int size) {
        double start = dg.start;
        double end= dg.end;
        double len = end - start;
        List<Double> output = new ArrayList<>(size);
        if (len == 0) {
            output.add(start);
        } else {
            double step = len / size;
            for (double s = start; s < end; s += step) {
                output.add(s);
            }
        }
        return output;
    }
}

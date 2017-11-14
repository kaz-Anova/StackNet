package com.stacknet.paramsearch.generators;

import java.util.List;

public interface Generator<T> {
    List<T> expend(int size);
}

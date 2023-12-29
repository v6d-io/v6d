/*
 * Copyright (C) 2007 The Guava Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.common.base;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.annotations.GwtCompatible;

import groovy.model.ValueHolder;

import java.util.StringJoiner;
import java.util.Arrays;

import javax.annotation.Nullable;

/**
 * Helper functions that can operate on any {@code Object}.
 *
 * @author Laurence Gonsalves
 * @since 2.0 (imported from Google Collections Library)
 */
@GwtCompatible
public final class Objects {
  private Objects() {}

  /**
   * Determines whether two possibly-null objects are equal. Returns:
   *
   * <ul>
   * <li>{@code true} if {@code a} and {@code b} are both null.
   * <li>{@code true} if {@code a} and {@code b} are both non-null and they are
   *     equal according to {@link Object#equals(Object)}.
   * <li>{@code false} in all other situations.
   * </ul>
   *
   * <p>This assumes that any non-null objects passed to this function conform
   * to the {@code equals()} contract.
   */
  public static boolean equal(@Nullable Object a, @Nullable Object b) {
    return a == b || (a != null && a.equals(b));
  }

  public static int hashCode(@Nullable Object... objects) {
    return Arrays.hashCode(objects);
  }

  /**
   * Creates an instance of {@link ToStringHelper}.
   *
   * <p>This is helpful for implementing {@link Object#toString()}.
   * Specification by example: <pre>   {@code
   *   // Returns "ClassName{}"
   *   Objects.toStringHelper(this)
   *       .toString();
   *
   *   // Returns "ClassName{x=1}"
   *   Objects.toStringHelper(this)
   *       .add("x", 1)
   *       .toString();
   *
   *   // Returns "MyObject{x=1}"
   *   Objects.toStringHelper("MyObject")
   *       .add("x", 1)
   *       .toString();
   *
   *   // Returns "ClassName{x=1, y=foo}"
   *   Objects.toStringHelper(this)
   *       .add("x", 1)
   *       .add("y", "foo")
   *       .toString();
   *   }}</pre>
   *
   * <p>Note that in GWT, class names are often obfuscated.
   *
   * @param self the object to generate the string for (typically {@code this}),
   *        used only for its class name
   * @since 2.0
   */
  public static ToStringHelper toStringHelper(Object self) {
    return new ToStringHelper(simpleName(self.getClass()));
  }

  /**
   * {@link Class#getSimpleName()} is not GWT compatible yet, so we
   * provide our own implementation.
   */
  private static String simpleName(Class<?> clazz) {
    String name = clazz.getName();

    // the nth anonymous class has a class name ending in "Outer$n"
    // and local inner classes have names ending in "Outer.$1Inner"
    name = name.replaceAll("\\$[0-9]+", "\\$");

    // we want the name of the inner class all by its lonesome
    int start = name.lastIndexOf('$');

    // if this isn't an inner class, just find the start of the
    // top level class name.
    if (start == -1) {
      start = name.lastIndexOf('.');
    }
    return name.substring(start + 1);
  }

  /**
   * Support class for {@link Objects#toStringHelper}.
   *
   * @author Jason Lee
   * @since 2.0
   */
  public static final class ToStringHelper {
    private final StringBuilder builder;
    private boolean needsSeparator = false;

    /**
     * Use {@link Objects#toStringHelper(Object)} to create an instance.
     */
    private ToStringHelper(String className) {
      checkNotNull(className);
      this.builder = new StringBuilder(32).append(className).append('{');
    }

    /**
     * Returns a string in the format specified by {@link
     * Objects#toStringHelper(Object)}.
     */
    @Override public String toString() {
      try {
        return builder.append('}').toString();
      } finally {
        // Slice off the closing brace in case there are additional calls to
        // #add or #addValue.
        builder.setLength(builder.length() - 1);
      }
    }

    public ToStringHelper add(String name, @Nullable Object value) {
      checkNameAndAppend(name).append(value);
      return this;
    }

    /**
     * Adds a name/value pair to the formatted output in {@code name=value}
     * format.
     *
     * @since 11.0 (source-compatible since 2.0)
     */
    public ToStringHelper add(String name, boolean value) {
      checkNameAndAppend(name).append(value);
      return this;
    }

    /**
     * Adds a name/value pair to the formatted output in {@code name=value}
     * format.
     *
     * @since 11.0 (source-compatible since 2.0)
     */
    public ToStringHelper add(String name, char value) {
      checkNameAndAppend(name).append(value);
      return this;
    }

    /**
     * Adds a name/value pair to the formatted output in {@code name=value}
     * format.
     *
     * @since 11.0 (source-compatible since 2.0)
     */
    public ToStringHelper add(String name, double value) {
      checkNameAndAppend(name).append(value);
      return this;
    }

    /**
     * Adds a name/value pair to the formatted output in {@code name=value}
     * format.
     *
     * @since 11.0 (source-compatible since 2.0)
     */
    public ToStringHelper add(String name, float value) {
      checkNameAndAppend(name).append(value);
      return this;
    }

    /**
     * Adds a name/value pair to the formatted output in {@code name=value}
     * format.
     *
     * @since 11.0 (source-compatible since 2.0)
     */
    public ToStringHelper add(String name, int value) {
      checkNameAndAppend(name).append(value);
      return this;
    }

    /**
     * Adds a name/value pair to the formatted output in {@code name=value}
     * format.
     *
     * @since 11.0 (source-compatible since 2.0)
     */
    public ToStringHelper add(String name, long value) {
      checkNameAndAppend(name).append(value);
      return this;
    }

    private StringBuilder checkNameAndAppend(String name) {
      checkNotNull(name);
      return maybeAppendSeparator().append(name).append('=');
    }

    private StringBuilder maybeAppendSeparator() {
      if (needsSeparator) {
        return builder.append(", ");
      } else {
        needsSeparator = true;
        return builder;
      }
    }
  }
}

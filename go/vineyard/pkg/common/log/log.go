/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package log contains utilities for fetching a new logger
// when one is not already available.
//
// # The Log Handle
//
// This package contains a root logr.Logger Log.  It may be used to
// get a handle to whatever the root logging implementation is.  By
// default, no implementation exists, and the handle returns "promises"
// to loggers.  When the implementation is set using SetLogger, these
// "promises" will be converted over to real loggers.
//
// # Logr
//
// All logging in controller-runtime is structured, using a set of interfaces
// defined by a package called logr
// (https://pkg.go.dev/github.com/go-logr/logr).  The sub-package zap provides
// helpers for setting up logr backed by Zap (go.uber.org/zap).
package log

import (
	"context"
	"fmt"
	"os"

	"github.com/go-logr/logr"
	"go.uber.org/zap/zapcore"

	"github.com/v6d-io/v6d/go/vineyard/pkg/common/log/zap"
)

var (
	defaultLogger = makeDefaultLogger(0)

	dlog = NewDelegatingLogSink(defaultLogger.GetSink())

	Log = Logger{logr.New(dlog).WithName("vineyard")}
)

func SetLogLevel(level int) {
	defaultLogger = makeDefaultLogger(level)
	dlog.Fulfill(defaultLogger.GetSink())
}

func makeDefaultLogger(verbose int) logr.Logger {
	zapOpts := &zap.Options{
		Development: true,
		TimeEncoder: zapcore.ISO8601TimeEncoder,
		Level:       zapcore.Level(verbose),
	}
	return zap.New(zap.UseOptions(zapOpts))
}

type Logger struct {
	logr.Logger
}

// SetLogger sets a concrete logging implementation for all deferred Loggers.
func SetLogger(l Logger) {
	dlog.Fulfill(l.GetSink())
}

// FromContext returns a logger with predefined values from a context.Context.
func FromContext(ctx context.Context, keysAndValues ...any) Logger {
	log := Log.Logger
	if ctx != nil {
		if logger, err := logr.FromContext(ctx); err == nil {
			log = logger
		}
	}
	return Logger{log.WithValues(keysAndValues...)}
}

// IntoContext takes a context and sets the logger as one of its values.
// Use FromContext function to retrieve the logger.
func IntoContext(ctx context.Context, log Logger) context.Context {
	return logr.NewContext(ctx, log.Logger)
}

func V(level int) Logger {
	return Logger{Log.V(level)}
}

func WithValues(keysAndValues ...any) Logger {
	return Logger{Log.WithValues(keysAndValues...)}
}

func WithName(name string) Logger {
	return Logger{Log.WithName(name)}
}

func (l Logger) Fatal(err error, msg string, keysAndValues ...any) {
	l.Error(err, msg, keysAndValues...)
	os.Exit(1)
}

func (l Logger) Infof(format string, v ...any) {
	l.Info(fmt.Sprintf(format, v...))
}

func (l Logger) Errorf(err error, format string, v ...any) {
	l.Error(err, fmt.Sprintf(format, v...))
}

func (l Logger) Fatalf(err error, format string, v ...any) {
	l.Fatal(err, fmt.Sprintf(format, v...))
}

func Info(msg string, keysAndValues ...any) {
	Log.Info(msg, keysAndValues...)
}

func Error(err error, msg string, keysAndValues ...any) {
	Log.Error(err, msg, keysAndValues...)
}

func Fatal(err error, msg string, keysAndValues ...any) {
	Log.Fatal(err, msg, keysAndValues...)
}

func Infof(format string, v ...any) {
	Log.Infof(format, v...)
}

func Errorf(err error, format string, v ...any) {
	Log.Errorf(err, format, v...)
}

func Fatalf(err error, format string, v ...any) {
	Log.Fatalf(err, format, v...)
}

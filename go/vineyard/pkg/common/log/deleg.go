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

package log

import (
	"sync"

	"github.com/go-logr/logr"
)

// loggerPromise knows how to populate a concrete logr.Logger
// with options, given an actual base logger later on down the line.
type loggerPromise struct {
	logger        *DelegatingLogSink
	childPromises []*loggerPromise
	promisesLock  sync.Mutex

	name *string
	tags []any
}

func (p *loggerPromise) WithName(l *DelegatingLogSink, name string) *loggerPromise {
	res := &loggerPromise{
		logger:       l,
		name:         &name,
		promisesLock: sync.Mutex{},
	}

	p.promisesLock.Lock()
	defer p.promisesLock.Unlock()
	p.childPromises = append(p.childPromises, res)
	return res
}

// WithValues provides a new Logger with the tags appended.
func (p *loggerPromise) WithValues(l *DelegatingLogSink, tags ...any) *loggerPromise {
	res := &loggerPromise{
		logger:       l,
		tags:         tags,
		promisesLock: sync.Mutex{},
	}

	p.promisesLock.Lock()
	defer p.promisesLock.Unlock()
	p.childPromises = append(p.childPromises, res)
	return res
}

// CallDepthLogSink represents a Logger that knows how to climb the call stack
// to identify the original call site and can offset the depth by a specified
// number of frames.  This is useful for users who have helper functions
// between the "real" call site and the actual calls to Logger methods.
// Implementations that log information about the call site (such as file,
// function, or line) would otherwise log information about the intermediate
// helper functions.
//
// This is an optional interface and implementations are not required to
// support it.
type CallDepthLogSink interface {
	// WithCallDepth returns a LogSink that will offset the call
	// stack by the specified number of frames when logging call
	// site information.
	//
	// If depth is 0, the LogSink should skip exactly the number
	// of call frames defined in RuntimeInfo.CallDepth when Info
	// or Error are called, i.e. the attribution should be to the
	// direct caller of Logger.Info or Logger.Error.
	//
	// If depth is 1 the attribution should skip 1 call frame, and so on.
	// Successive calls to this are additive.
	WithCallDepth(depth int) logr.Logger
}

// Fulfill instantiates the Logger with the provided logger.
func (p *loggerPromise) Fulfill(parentLogSink logr.Logger) {
	sink := parentLogSink
	if p.name != nil {
		sink = sink.WithName(*p.name)
	}

	if p.tags != nil {
		sink = sink.WithValues(p.tags...)
	}

	p.logger.lock.Lock()
	p.logger.logger = sink
	if withCallDepth, ok := sink.(CallDepthLogSink); ok {
		p.logger.logger = withCallDepth.WithCallDepth(1)
	}
	p.logger.promise = nil
	p.logger.lock.Unlock()

	for _, childPromise := range p.childPromises {
		childPromise.Fulfill(sink)
	}
}

// RuntimeInfo holds information that the logr "core" library knows which
// LogSinks might want to know.
type RuntimeInfo struct {
	// CallDepth is the number of call frames the logr library adds between the
	// end-user and the LogSink.  LogSink implementations which choose to print
	// the original logging site (e.g. file & line) should climb this many
	// additional frames to find it.
	CallDepth int
}

// runtimeInfo is a static global.  It must not be changed at run time.
var runtimeInfo = RuntimeInfo{
	CallDepth: 1,
}

// DelegatingLogSink is a logsink that delegates to another logr.LogSink.
// If the underlying promise is not nil, it registers calls to sub-loggers with
// the logging factory to be populated later, and returns a new delegating
// logger.  It expects to have *some* logr.Logger set at all times (generally
// a no-op logger before the promises are fulfilled).
type DelegatingLogSink struct {
	lock    sync.RWMutex
	logger  logr.Logger
	promise *loggerPromise
	info    RuntimeInfo
}

// Init implements logr.LogSink.
func (l *DelegatingLogSink) Init(info RuntimeInfo) {
	l.lock.Lock()
	defer l.lock.Unlock()
	l.info = info
}

// Enabled tests whether this Logger is enabled.  For example, commandline
// flags might be used to set the logging verbosity and disable some info
// logs.
func (l *DelegatingLogSink) Enabled() bool {
	l.lock.RLock()
	defer l.lock.RUnlock()
	return l.logger.Enabled()
}

// Info logs a non-error message with the given key/value pairs as context.
//
// The msg argument should be used to add some constant description to
// the log line.  The key/value pairs can then be used to add additional
// variable information.  The key/value pairs should alternate string
// keys and arbitrary values.
func (l *DelegatingLogSink) Info(msg string, keysAndValues ...any) {
	l.lock.RLock()
	defer l.lock.RUnlock()
	l.logger.Info(msg, keysAndValues...)
}

// Error logs an error, with the given message and key/value pairs as context.
// It functions similarly to calling Info with the "error" named value, but may
// have unique behavior, and should be preferred for logging errors (see the
// package documentations for more information).
//
// The msg field should be used to add context to any underlying error,
// while the err field should be used to attach the actual error that
// triggered this log line, if present.
func (l *DelegatingLogSink) Error(err error, msg string, keysAndValues ...any) {
	l.lock.RLock()
	defer l.lock.RUnlock()
	l.logger.Error(err, msg, keysAndValues...)
}

func (l *DelegatingLogSink) V(level int) logr.Logger {
	l.lock.RLock()
	defer l.lock.RUnlock()
	return l.logger.V(level)
}

// WithName provides a new Logger with the name appended.
func (l *DelegatingLogSink) WithName(name string) logr.Logger {
	l.lock.RLock()
	defer l.lock.RUnlock()

	if l.promise == nil {
		sink := l.logger.WithName(name)
		if withCallDepth, ok := sink.(CallDepthLogSink); ok {
			sink = withCallDepth.WithCallDepth(-1)
		}
		return sink
	}

	res := &DelegatingLogSink{logger: l.logger}
	promise := l.promise.WithName(res, name)
	res.promise = promise

	return res
}

// WithValues provides a new Logger with the tags appended.
func (l *DelegatingLogSink) WithValues(tags ...any) logr.Logger {
	l.lock.RLock()
	defer l.lock.RUnlock()

	if l.promise == nil {
		sink := l.logger.WithValues(tags...)
		if withCallDepth, ok := sink.(CallDepthLogSink); ok {
			sink = withCallDepth.WithCallDepth(-1)
		}
		return sink
	}

	res := &DelegatingLogSink{logger: l.logger}
	promise := l.promise.WithValues(res, tags...)
	res.promise = promise

	return res
}

// Fulfill switches the logger over to use the actual logger
// provided, instead of the temporary initial one, if this method
// has not been previously called.
func (l *DelegatingLogSink) Fulfill(actual logr.Logger) {
	if l.promise != nil {
		l.promise.Fulfill(actual)
	}
}

// NewDelegatingLogSink constructs a new DelegatingLogSink which uses
// the given logger before its promise is fulfilled.
func NewDelegatingLogSink(initial logr.Logger) *DelegatingLogSink {
	l := &DelegatingLogSink{
		logger:  initial,
		promise: &loggerPromise{promisesLock: sync.Mutex{}},
	}
	l.promise.logger = l
	return l
}

# ClearCL #

[![Join the chat at https://gitter.im/ClearCL/Lobby](https://badges.gitter.im/ClearCL/Lobby.svg)](https://gitter.im/ClearCL/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Multi-backend Java Object Oriented Facade API for OpenCL. 

## Why?

OpenCL libraries come and go in Java, some are great but then one day the lead developper goes on to greener pastures and you are left with code that needs to be rewritten to take advantage of a new up-to-date library with better support. Maybe a particular library has a bug or does not support the function you need? or it does not give you access to the underlying native pointers, making difficult to process large buffers/images or interoperate with hardware? or maybe it just does not support your exotic OS of choice. To protect your code from complete rewrites ClearCL offers a very clean and complete API to write your code against. Changing backend requires just changing one line of code.   

ClearVolume 2.0 GPU code will be built on top of ClearCL to offer flexibility and robsutness against OpenCL library idiosyncrasies and eventual deprecation.

## Features:
1. Implemented backends: JOCL (www.jocl.org/) and JavaCL (github.com/nativelibs4java/JavaCL).
2. Full support of OpenCL 1.2
3. Support for offheap memory (> 2G) via CoreMem (http://github.com/ClearControl/CoreMem)
4. Automatic backend selection (different backends works better on some platforms).
5. Automatic device selection via kernel benchmarking.
6. Supports OpenCL 1.0/1.1 devices by automatically using alternative functions.

## In progress:
1. Full support for events
2. Improve robustness: ClearCL is used the current ClearVolume ImageJ plugin.

## Planned features:
1. Basic set of OpenCL kernels for image processing: denoising, deconvolution, image quality, correlation, projection...
2. Scatter-gather for processing images and buffers that don't fit in GPU mem.
3. Upload-and-scaledown functionality to load and scale down images into GPU memory
4. Live-coding infrastructire to be able to edit kernel code and immediately see the result.
5. Integration with imglib2 and FiJi.

## How to add ClearCL as a dependency to your project:

Find the latest version on [BinTray](https://bintray.com/clearvolume/ClearVolume/ClearCL)

### With Gradle:
~~~~
     compile 'net.clearvolume:clearcl:0.3.36'
~~~~

~~~~
repositories {
    maven {
        url  "http://dl.bintray.com/clearvolume/ClearVolume" 
    }
}
~~~~

### With Maven:
~~~~
<dependency>
  <groupId>net.clearvolume</groupId>
  <artifactId>clearcl</artifactId>
  <version>0.3.36</version>
  <type>pom</type>
</dependency>
~~~~

~~~~
<repository>
     <snapshots>
         <enabled>false</enabled>
     </snapshots>
     <id>bintray-clearvolume-ClearVolume</id>
     <name>bintray</name>
     <url>http://dl.bintray.com/clearvolume/ClearVolume</url>
 </repository>
~~~~



## Getting started:

Just check the test (here)[https://github.com/ClearVolume/ClearCL/blob/master/src/java/clearcl/test/ClearCLTests.java] to learn how to use ClearCL. More tests are coming...

## How to build project with Gradle

1. Clone the project
2. run the Gradle Wrapper that comes with the repo:
~~~~ 
     ./gradlew cleanEclipse eclipse build 
~~~~
     
~~~~
     ./gradlew idea build 
~~~~

## Internals & how to implement backends:

Implementing backends simply consists in implementing classes against this [interface](https://github.com/ClearVolume/ClearCL/blob/master/src/java/clearcl/backend/ClearCLBackendInterface.java).

OpenCL binding libraries such as (or wthin) JOCL, JavaCL, Jogamp, and LWJGL encapsulate native pointers/handles
using specific classes. ClearCL backends further encapsulate these within [ClearCLPeerPointers](https://github.com/ClearVolume/ClearCL/blob/master/src/java/clearcl/ClearCLPeerPointer.java). This pointer wrapper class is not exposed by the Object Oriented API but instead is only used from within the backend implementations and within the OO classes.

## Contributors

* Loic Royer ( royer -at- mpi-cbg -point- de )
* you?

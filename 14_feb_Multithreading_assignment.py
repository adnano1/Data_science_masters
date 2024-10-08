{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "collapsed_sections": [
        "oskUvuoLMp96",
        "sbLS0dksNgoQ"
      ]
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "###Q.1) What is multithreading in python?\n",
        "* Multithreading in Python refers to the ability of a program to execute multiple threads of execution concurrently within the same process. Each thread runs independently and can perform its own task while sharing the same memory space and resources with the other threads.\n",
        "\n",
        "###Why is it used? \n",
        "* Multithreading is used to improve the performance of a program by allowing it to execute multiple tasks concurrently, thus utilizing the available resources efficiently. For example, in a web server, multiple clients can be served simultaneously by creating a separate thread for each client request, rather than serving them one at a time.\n",
        "\n",
        "###Name the module used to handle threads in python\n",
        "* The threading module is used to handle threads in Python.  It provides a simple way to create, start, pause, and terminate threads in a Python program. The module also includes synchronization primitives, such as locks and semaphores, to help prevent multiple threads from accessing the same resource at the same time, which can cause race conditions and other synchronization problems.\n"
      ],
      "metadata": {
        "id": "oskUvuoLMp96"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "fFGXa9YwMomL"
      },
      "outputs": [],
      "source": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Q.2) Why threading module used? \n",
        "\n",
        "#####The threading module in Python is used to create and manage threads of execution within a single process. Here are some of the reasons why threading module is used:\n",
        "\n",
        "* To improve program performance: Multithreading allows multiple tasks to be performed concurrently, thus improving the performance of a program.\n",
        "\n",
        "* To make efficient use of resources: By utilizing the available resources efficiently, multithreading allows programs to make more efficient use of system resources, such as CPU time and memory.\n",
        "\n",
        "* To achieve concurrency: Threading allows multiple parts of a program to execute concurrently, thus enabling the program to be more responsive to user input.\n",
        "\n",
        "* To avoid blocking: By executing time-consuming tasks in a separate thread, the main thread of execution can continue to respond to user input and other events, without being blocked.\n",
        "\n",
        "### write the use of the following functions\n",
        " ## activeCount\n",
        " * activeCount(): The activeCount() function is a method of the threading module in Python that is used to get the number of Thread objects that are currently active and running in a Python program. This function returns an integer that represents the number of active threads.\n",
        "* For example, you can use this function to check the number of threads currently running in your program and use this information to optimize your program's performance.\n",
        " ## currentThread\n",
        " * currentThread(): The currentThread() function is a method of the threading module in Python that is used to get a reference to the current Thread object that is executing the current code. This function returns the Thread object that represents the currently executing thread.\n",
        "* This function is useful for debugging and for passing the current thread object as a parameter to other functions.\n",
        " ## enumerate\n",
        " * enumerate(): The enumerate() function is a method of the threading module in Python that is used to get a list of all Thread objects that are currently active and running in a Python program. This function returns a list of Thread objects.\n",
        "* For example, you can use this function to get a list of all the threads currently running in your program and use this information to monitor the progress of your program or to terminate threads that are no longer needed. This function can be combined with other threading functions to create powerful multithreaded applications."
      ],
      "metadata": {
        "id": "sbLS0dksNgoQ"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "###Q.3) Explain the following functions\n",
        "### run\n",
        "* run(): The run() method is a method of the Thread class in Python, which is used to define the behavior of the thread when it starts running. This method contains the code that will be executed when the thread is started.\n",
        " * The run() method should be overridden in a subclass of the Thread class to define the behavior of the thread. When the thread is started, the run() method of the subclass will be executed in a separate thread.\n",
        " ### start\n",
        "* start(): The start() method is a method of the Thread class in Python that is used to start a new thread of execution. When this method is called, a new thread is created and the run() method of the thread is executed.\n",
        " * The start() method should be called only once for each thread object. Calling it more than once will raise a RuntimeError.\n",
        " \n",
        " ## join\n",
        "* join(): The join() method is a method of the Thread class in Python that is used to wait for a thread to finish executing. When this method is called, the calling thread will wait until the thread being joined has finished executing.\n",
        " * The join() method can be called with a timeout argument, which specifies the maximum amount of time that the calling thread should wait for the joined thread to finish. If the joined thread does not finish executing within the specified timeout, the join() method will return and the calling thread can continue executing.\n",
        "###  isAlive\n",
        "* isAlive(): The isAlive() method is a method of the Thread class in Python that is used to check whether a thread is currently executing. When this method is called, it returns a Boolean value that indicates whether the thread is currently executing (True) or not (False).\n",
        " * This method is useful for checking the status of a thread and determining whether it has finished executing or not. The isAlive() method can be used in combination with the join() method to wait for a thread to finish executing before continuing with the main program."
      ],
      "metadata": {
        "id": "Gx4s8MylPxl1"
      }
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "DBTwQ2UtQw1s"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "###Q.4) Write a python program to create two threads. Thread one must print the list of squares and thread two must print the list of cubes"
      ],
      "metadata": {
        "id": "oyHlksd1Q0Lv"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import threading\n",
        "\n",
        "def print_squares():\n",
        "    for i in range(1, 11):\n",
        "        print(i ** 2)\n",
        "\n",
        "def print_cubes():\n",
        "    for i in range(1, 11):\n",
        "        print(i ** 3)\n",
        "\n",
        "# Create two threads\n",
        "t1 = threading.Thread(target=print_squares)\n",
        "t2 = threading.Thread(target=print_cubes)\n",
        "\n",
        "# Start the threads\n",
        "t1.start()\n",
        "t2.start()\n",
        "\n",
        "# Wait for the threads to finish\n",
        "t1.join()\n",
        "t2.join()\n",
        "\n",
        "print(\"Done!\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-O46bhnzRP3i",
        "outputId": "24ff4770-bfa9-43f6-ba96-9634f11e4b83"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1\n",
            "4\n",
            "9\n",
            "16\n",
            "25\n",
            "36\n",
            "49\n",
            "64\n",
            "81\n",
            "100\n",
            "1\n",
            "8\n",
            "27\n",
            "64\n",
            "125\n",
            "216\n",
            "343\n",
            "512\n",
            "729\n",
            "1000\n",
            "Done!\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "###Q.5) State advantages and disadvantages of multithreading\n",
        "\n",
        "####Here are some advantages and disadvantages of multithreading:\n",
        "\n",
        "##Advantages:\n",
        "\n",
        "* Improved performance: Multithreading allows a program to utilize multiple CPUs or CPU cores, which can improve the overall performance of the program.\n",
        "\n",
        "* Enhanced responsiveness: Multithreading allows a program to remain responsive even when performing long-running tasks. By executing these tasks in a separate thread, the main thread can continue to handle user input and respond to events.\n",
        "\n",
        "* Simplified design: Multithreading can simplify the design of a program by allowing different parts of the program to run concurrently without the need for complex coordination between them.\n",
        "\n",
        "* Resource sharing: Multithreading allows threads to share resources, such as memory and files which can reduce the overall resource usage of the program.\n",
        "\n",
        "## Disadvantages:\n",
        "\n",
        "* Complexity: Multithreading can make a program more complex and harder to debug due to issues such as race conditions, deadlocks, and synchronization.\n",
        "\n",
        "* Overhead: Multithreading requires additional overhead in terms of memory and CPU usage, which can reduce the overall performance of the program.\n",
        "\n",
        "* Synchronization: Multithreading requires careful synchronization of shared resources to prevent issues such as data corruption and race conditions.\n",
        "\n",
        "*Debugging: Debugging a multithreaded program can be more challenging than debugging a single-threaded program due to the increased complexity and potential for race conditions and synchronization issues."
      ],
      "metadata": {
        "id": "3BSZ_XSfYz2o"
      }
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "xL6MBrWfRVnn"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "###Q6) Explain deadlocks and race conditions.\n",
        "\n",
        "## Deadlocks:\n",
        "* A deadlock occurs when two or more threads are blocked waiting for each other to release resources. This can happen when each thread holds a resource that another thread needs to continue executing, and neither thread is able to proceed. Deadlocks can cause a program to freeze or become unresponsive, and can be difficult to detect and fix.\n",
        "\n",
        " * For example, imagine that Thread 1 holds a lock on Resource A and is waiting for Resource B, while Thread 2 holds a lock on Resource B and is waiting for Resource A. In this scenario, both threads are blocked waiting for the other thread to release its resource, resulting in a deadlock.\n",
        "\n",
        "## Race conditions:\n",
        "* A race condition occurs when multiple threads access and modify a shared resource concurrently, and the outcome depends on the order in which the threads execute. This can cause unpredictable and incorrect behavior, as the result of the program can vary depending on timing and scheduling.\n",
        "\n",
        "* For example, imagine that two threads are updating a shared counter variable. Thread 1 reads the value of the counter, increments it, and writes the new value back to the counter. Thread 2 does the same thing, but in the opposite order: it reads the value of the counter, increments it, and writes the new value back to the counter. If both threads execute concurrently and read the same value of the counter, they will both increment it and write the same value back, effectively only incrementing the counter once instead of twice."
      ],
      "metadata": {
        "id": "9dY89nKUa-Lf"
      }
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "lXTGysQTcGhE"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}

#pragma once

// Minimal GLFW 3 ABI declarations used by this project.  This fallback lets
// CMake link the GLFW shared library bundled by the Python glfw package when
// the distro development headers are unavailable.  A system GLFW header is
// preferred whenever glfw3 is installed normally.

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GLFWwindow GLFWwindow;
typedef struct GLFWmonitor GLFWmonitor;

typedef void (*GLFWkeyfun)(GLFWwindow*, int, int, int, int);
typedef void (*GLFWmousebuttonfun)(GLFWwindow*, int, int, int);
typedef void (*GLFWcursorposfun)(GLFWwindow*, double, double);
typedef void (*GLFWscrollfun)(GLFWwindow*, double, double);

#define GLFW_FALSE 0
#define GLFW_TRUE 1
#define GLFW_PRESS 1
#define GLFW_REPEAT 2

#define GLFW_MOUSE_BUTTON_LEFT 0
#define GLFW_MOUSE_BUTTON_RIGHT 1
#define GLFW_MOUSE_BUTTON_MIDDLE 2

#define GLFW_KEY_SPACE 32
#define GLFW_KEY_A 65
#define GLFW_KEY_D 68
#define GLFW_KEY_R 82
#define GLFW_KEY_S 83
#define GLFW_KEY_W 87
#define GLFW_KEY_X 88
#define GLFW_KEY_ESCAPE 256
#define GLFW_KEY_LEFT_SHIFT 340
#define GLFW_KEY_RIGHT_SHIFT 344

int glfwInit(void);
void glfwTerminate(void);
GLFWwindow* glfwCreateWindow(int width, int height, const char* title,
                             GLFWmonitor* monitor, GLFWwindow* share);
void glfwDestroyWindow(GLFWwindow* window);
void glfwMakeContextCurrent(GLFWwindow* window);
void glfwSwapInterval(int interval);
void glfwSwapBuffers(GLFWwindow* window);
void glfwPollEvents(void);
int glfwWindowShouldClose(GLFWwindow* window);
void glfwSetWindowShouldClose(GLFWwindow* window, int value);
void glfwSetWindowUserPointer(GLFWwindow* window, void* pointer);
void* glfwGetWindowUserPointer(GLFWwindow* window);
GLFWkeyfun glfwSetKeyCallback(GLFWwindow* window, GLFWkeyfun callback);
GLFWmousebuttonfun glfwSetMouseButtonCallback(GLFWwindow* window,
                                              GLFWmousebuttonfun callback);
GLFWcursorposfun glfwSetCursorPosCallback(GLFWwindow* window,
                                          GLFWcursorposfun callback);
GLFWscrollfun glfwSetScrollCallback(GLFWwindow* window,
                                    GLFWscrollfun callback);
int glfwGetMouseButton(GLFWwindow* window, int button);
int glfwGetKey(GLFWwindow* window, int key);
void glfwGetCursorPos(GLFWwindow* window, double* xpos, double* ypos);
void glfwGetWindowSize(GLFWwindow* window, int* width, int* height);
void glfwGetFramebufferSize(GLFWwindow* window, int* width, int* height);

#ifdef __cplusplus
}
#endif

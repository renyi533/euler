#include <stdlib.h>
#include <dlfcn.h>
#include <stdio.h>

int main (int argc, char *argv[])
{
  if (argc < 2) {
    printf("please specify euler tf wrapper lib\n");
    return -1;
  }
  
  char* tf_euler_lib = argv[1];
  
  printf("tf_euler_lib:%s\n", tf_euler_lib);
  
  
  void* tf_wrapper_handle = dlopen(tf_euler_lib, RTLD_NOW | RTLD_GLOBAL);
  if (tf_wrapper_handle == NULL) {
    printf("dlopen - %s\n", dlerror());
    exit(-1);
  }
  bool (*create_graph_func)(const char *);
  create_graph_func = (bool (*)(const char*))dlsym(tf_wrapper_handle, "CreateGraph");
  if (create_graph_func == NULL) {
    printf("dlsym - %s\n", dlerror());
    exit(-1);
  }
  (*create_graph_func)("mode=Remote;zk_server=127.0.0.1:2801;zk_path=/euler");
  dlclose(tf_wrapper_handle);
  return 0;
}

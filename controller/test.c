#include <stdio.h>
#include <hexagon_nn.h>

int main(){
  int version;
  hexagon_nn_version(&version);
  printf("HexagonNN version: %d\n", version);
  printf("Success!\n");

  return 0;
}

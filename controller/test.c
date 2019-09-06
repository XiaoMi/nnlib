#include <stdio.h>
#include <hexagon_nn.h>

int main(){
  int version;
  if (hexagon_nn_version(&version) != 0) {
    printf("Failed.\n");
    return 0;
  }
  printf("HexagonNN version: %d\n", version);
  printf("Success!\n");

  return 0;
}

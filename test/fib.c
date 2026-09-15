#include <stdio.h>

unsigned long long fib(unsigned int n) {
    if (n <= 1) return n;
    return fib(n - 1) + fib(n - 2);
}

int main(int argc, char* argv[]) {
    unsigned int n = 40; 
    
    unsigned long long result = fib(n);
    
    printf("Fibonacci(%u) = %llu\n", n, result);
    
    return 0;
}

#include <cstdio>
#include <windows.h>
int main() {
    HMODULE h = LoadLibraryW(L"openvino.dll");
    if (!h) { printf("openvino.dll not found\n"); return 1; }
    HMODULE hc = LoadLibraryW(L"openvino_c.dll");
    printf("openvino.dll   ov_tensor_create_from_host_ptr = %p\n",
           GetProcAddress(h, "ov_tensor_create_from_host_ptr"));
    if (hc) printf("openvino_c.dll ov_tensor_create_from_host_ptr = %p\n",
                   GetProcAddress(hc, "ov_tensor_create_from_host_ptr"));
    return 0;
}

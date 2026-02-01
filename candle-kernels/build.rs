use cudaforge::KernelBuilder;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/compatibility.cuh");
    println!("cargo:rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo:rerun-if-changed=src/binary_op_macros.cuh");

    let mut bindings = KernelBuilder::new()
        .source_dir("src") // Scan src/ for .cu files
        .arg("-fmad=false")       // Disable FMA rounding non-determinism
        .arg("-ftz=false")        // Preserve subnormals (critical for attention)
        .build_ptx()
        .expect("Failed to compile CUDA kernels")
        .write("src/lib.rs")
        .expect("Failed to write PTX bindings");
//        .arg("-prec-div")         // Enable precise division
//        .arg("-prec-sqrt")        // Enable precise square-root

}

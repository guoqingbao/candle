use anyhow::{Context, Result};
use cudaforge::KernelBuilder;
use std::path::PathBuf;

fn main() -> Result<()> {
    let flash_decoding_enabled = std::env::var("CARGO_FEATURE_FLASH_DECODING").is_ok();
    let flash_context_enabled = std::env::var("CARGO_FEATURE_FLASH_CONTEXT").is_ok();

    let mut kernel_files: Vec<&str> = if !flash_context_enabled {
        vec![
            "kernels/flash_api.cu",
            "kernels/flash_fwd_hdim128_fp16_causal_sm80.cu",
            "kernels/flash_fwd_hdim160_fp16_causal_sm80.cu",
            "kernels/flash_fwd_hdim192_fp16_causal_sm80.cu",
            "kernels/flash_fwd_hdim256_fp16_causal_sm80.cu",
            "kernels/flash_fwd_hdim32_fp16_causal_sm80.cu",
            "kernels/flash_fwd_hdim64_fp16_causal_sm80.cu",
            "kernels/flash_fwd_hdim96_fp16_causal_sm80.cu",
            "kernels/flash_fwd_hdim128_bf16_causal_sm80.cu",
            "kernels/flash_fwd_hdim160_bf16_causal_sm80.cu",
            "kernels/flash_fwd_hdim192_bf16_causal_sm80.cu",
            "kernels/flash_fwd_hdim256_bf16_causal_sm80.cu",
            "kernels/flash_fwd_hdim32_bf16_causal_sm80.cu",
            "kernels/flash_fwd_hdim64_bf16_causal_sm80.cu",
            "kernels/flash_fwd_hdim96_bf16_causal_sm80.cu",
        ]
    } else {
        vec![
            "kernels/flash_api.cu",
            "kernels/flash_fwd_hdim128_fp16_causal_sm80.cu",
            "kernels/flash_fwd_hdim64_fp16_causal_sm80.cu",
            "kernels/flash_fwd_hdim128_bf16_causal_sm80.cu",
            "kernels/flash_fwd_hdim64_bf16_causal_sm80.cu",
        ]
    };

    if flash_context_enabled {
        kernel_files.extend_from_slice(&[
            "kernels/flash_fwd_split_hdim128_fp16_sm80.cu",
            "kernels/flash_fwd_split_hdim64_fp16_sm80.cu",
            "kernels/flash_fwd_split_hdim128_bf16_sm80.cu",
            "kernels/flash_fwd_split_hdim64_bf16_sm80.cu",
        ]);
    } else if flash_decoding_enabled {
        kernel_files.extend_from_slice(&[
            "kernels/flash_fwd_split_hdim128_fp16_sm80.cu",
            "kernels/flash_fwd_split_hdim160_fp16_sm80.cu",
            "kernels/flash_fwd_split_hdim192_fp16_sm80.cu",
            "kernels/flash_fwd_split_hdim256_fp16_sm80.cu",
            "kernels/flash_fwd_split_hdim32_fp16_sm80.cu",
            "kernels/flash_fwd_split_hdim64_fp16_sm80.cu",
            "kernels/flash_fwd_split_hdim96_fp16_sm80.cu",
            "kernels/flash_fwd_split_hdim128_bf16_sm80.cu",
            "kernels/flash_fwd_split_hdim160_bf16_sm80.cu",
            "kernels/flash_fwd_split_hdim192_bf16_sm80.cu",
            "kernels/flash_fwd_split_hdim256_bf16_sm80.cu",
            "kernels/flash_fwd_split_hdim32_bf16_sm80.cu",
            "kernels/flash_fwd_split_hdim64_bf16_sm80.cu",
            "kernels/flash_fwd_split_hdim96_bf16_sm80.cu",
        ]);
    }

    println!("cargo:rerun-if-changed=build.rs");
    for kernel_file in &kernel_files {
        println!("cargo:rerun-if-changed={}", kernel_file);
    }

    println!("cargo:rerun-if-changed=kernels/flash_fwd_kernel.h");
    println!("cargo:rerun-if-changed=kernels/flash_fwd_launch_template.h");
    println!("cargo:rerun-if-changed=kernels/flash.h");
    println!("cargo:rerun-if-changed=kernels/philox.cuh");
    println!("cargo:rerun-if-changed=kernels/softmax.h");
    println!("cargo:rerun-if-changed=kernels/utils.h");
    println!("cargo:rerun-if-changed=kernels/kernel_traits.h");
    println!("cargo:rerun-if-changed=kernels/block_info.h");
    println!("cargo:rerun-if-changed=kernels/static_switch.h");
    println!("cargo:rerun-if-changed=kernels/hardware_info.h");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").context("OUT_DIR not set")?);
    let build_dir = match std::env::var("CANDLE_FLASH_ATTN_BUILD_DIR") {
        Err(_) => out_dir.clone(),
        Ok(build_dir) => {
            let path = PathBuf::from(build_dir);
            path.canonicalize().unwrap_or_else(|_| {
                panic!(
                    "Directory doesn't exist: {} (the current directory is {})",
                    path.display(),
                    std::env::current_dir().unwrap().display()
                )
            })
        }
    };

    let mut builder = KernelBuilder::new()
        .source_files(&kernel_files)
        .out_dir(&build_dir)
        .with_cutlass(None) // ✅ Auto-fetch CUTLASS from GitHub
        .arg("-O3")
        .arg("-fmad=false")       // Disable FMA rounding non-determinism
        .arg("-ftz=false")        // Preserve subnormals (critical for attention)
        .arg("-prec-div=true")         // Enable precise division
        .arg("-prec-sqrt=true")        // Enable precise square-root
        .arg("-std=c++17")
        .arg("-U__CUDA_NO_HALF_OPERATORS__")
        .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
        .arg("-U__CUDA_NO_HALF2_OPERATORS__")
        .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("--verbose")
        .arg("-Xfatbin")
        .arg("-compress-all")
        .arg("-Xcompiler")
        .arg("-fPIC");

    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            builder = builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    if flash_decoding_enabled {
        builder = builder.arg("-DFLASH_DECODING");
    }

    if flash_context_enabled {
        builder = builder.arg("-DFLASH_CONTEXT");
    }

    builder.build_lib(build_dir.join("libflashattention.a"))?;

    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=flashattention");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    Ok(())
}

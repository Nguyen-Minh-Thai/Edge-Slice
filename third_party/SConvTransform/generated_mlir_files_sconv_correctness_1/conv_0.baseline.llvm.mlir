module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @conv_2d_nchw_fchw(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: !llvm.ptr, %arg23: !llvm.ptr, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: i64, %arg28: i64, %arg29: i64, %arg30: i64, %arg31: i64, %arg32: i64) -> !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %1 = llvm.insertvalue %arg22, %0[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %2 = llvm.insertvalue %arg23, %1[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %3 = llvm.insertvalue %arg24, %2[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %4 = llvm.insertvalue %arg25, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %5 = llvm.insertvalue %arg29, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %6 = llvm.insertvalue %arg26, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %7 = llvm.insertvalue %arg30, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %8 = llvm.insertvalue %arg27, %7[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %9 = llvm.insertvalue %arg31, %8[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %10 = llvm.insertvalue %arg28, %9[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %11 = llvm.insertvalue %arg32, %10[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %12 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %13 = llvm.insertvalue %arg11, %12[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %14 = llvm.insertvalue %arg12, %13[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %15 = llvm.insertvalue %arg13, %14[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %16 = llvm.insertvalue %arg14, %15[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %17 = llvm.insertvalue %arg18, %16[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %18 = llvm.insertvalue %arg15, %17[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %19 = llvm.insertvalue %arg19, %18[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %20 = llvm.insertvalue %arg16, %19[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %21 = llvm.insertvalue %arg20, %20[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %22 = llvm.insertvalue %arg17, %21[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %23 = llvm.insertvalue %arg21, %22[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %24 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %25 = llvm.insertvalue %arg0, %24[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %26 = llvm.insertvalue %arg1, %25[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %27 = llvm.insertvalue %arg2, %26[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %28 = llvm.insertvalue %arg3, %27[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %29 = llvm.insertvalue %arg7, %28[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %30 = llvm.insertvalue %arg4, %29[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %31 = llvm.insertvalue %arg8, %30[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %32 = llvm.insertvalue %arg5, %31[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %33 = llvm.insertvalue %arg9, %32[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %34 = llvm.insertvalue %arg6, %33[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %35 = llvm.insertvalue %arg10, %34[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %36 = llvm.mlir.constant(0 : index) : i64
    %37 = llvm.mlir.constant(1 : index) : i64
    %38 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%36 : i64)
  ^bb1(%39: i64):  // 2 preds: ^bb0, ^bb20
    %40 = llvm.icmp "slt" %39, %37 : i64
    llvm.cond_br %40, ^bb2, ^bb21
  ^bb2:  // pred: ^bb1
    %41 = llvm.mlir.constant(0 : index) : i64
    %42 = llvm.mlir.constant(16 : index) : i64
    %43 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb3(%41 : i64)
  ^bb3(%44: i64):  // 2 preds: ^bb2, ^bb19
    %45 = llvm.icmp "slt" %44, %42 : i64
    llvm.cond_br %45, ^bb4, ^bb20
  ^bb4:  // pred: ^bb3
    %46 = llvm.mlir.constant(0 : index) : i64
    %47 = llvm.mlir.constant(8 : index) : i64
    %48 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb5(%46 : i64)
  ^bb5(%49: i64):  // 2 preds: ^bb4, ^bb18
    %50 = llvm.icmp "slt" %49, %47 : i64
    llvm.cond_br %50, ^bb6, ^bb19
  ^bb6:  // pred: ^bb5
    %51 = llvm.mlir.constant(0 : index) : i64
    %52 = llvm.mlir.constant(8 : index) : i64
    %53 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb7(%51 : i64)
  ^bb7(%54: i64):  // 2 preds: ^bb6, ^bb17
    %55 = llvm.icmp "slt" %54, %52 : i64
    llvm.cond_br %55, ^bb8, ^bb18
  ^bb8:  // pred: ^bb7
    %56 = llvm.mlir.constant(0 : index) : i64
    %57 = llvm.mlir.constant(8 : index) : i64
    %58 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb9(%56 : i64)
  ^bb9(%59: i64):  // 2 preds: ^bb8, ^bb16
    %60 = llvm.icmp "slt" %59, %57 : i64
    llvm.cond_br %60, ^bb10, ^bb17
  ^bb10:  // pred: ^bb9
    %61 = llvm.mlir.constant(0 : index) : i64
    %62 = llvm.mlir.constant(3 : index) : i64
    %63 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb11(%61 : i64)
  ^bb11(%64: i64):  // 2 preds: ^bb10, ^bb15
    %65 = llvm.icmp "slt" %64, %62 : i64
    llvm.cond_br %65, ^bb12, ^bb16
  ^bb12:  // pred: ^bb11
    %66 = llvm.mlir.constant(0 : index) : i64
    %67 = llvm.mlir.constant(3 : index) : i64
    %68 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb13(%66 : i64)
  ^bb13(%69: i64):  // 2 preds: ^bb12, ^bb14
    %70 = llvm.icmp "slt" %69, %67 : i64
    llvm.cond_br %70, ^bb14, ^bb15
  ^bb14:  // pred: ^bb13
    %71 = llvm.add %49, %64 : i64
    %72 = llvm.add %54, %69 : i64
    %73 = llvm.extractvalue %35[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %74 = llvm.extractvalue %35[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %75 = llvm.getelementptr %73[%74] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %76 = llvm.extractvalue %35[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %77 = llvm.mul %39, %76 : i64
    %78 = llvm.extractvalue %35[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %79 = llvm.mul %59, %78 : i64
    %80 = llvm.add %77, %79 : i64
    %81 = llvm.extractvalue %35[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %82 = llvm.mul %71, %81 : i64
    %83 = llvm.add %80, %82 : i64
    %84 = llvm.extractvalue %35[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %85 = llvm.mul %72, %84 : i64
    %86 = llvm.add %83, %85 : i64
    %87 = llvm.getelementptr %75[%86] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %88 = llvm.load %87 : !llvm.ptr -> f32
    %89 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %90 = llvm.extractvalue %23[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %91 = llvm.getelementptr %89[%90] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %92 = llvm.extractvalue %23[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %93 = llvm.mul %44, %92 : i64
    %94 = llvm.extractvalue %23[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %95 = llvm.mul %59, %94 : i64
    %96 = llvm.add %93, %95 : i64
    %97 = llvm.extractvalue %23[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %98 = llvm.mul %64, %97 : i64
    %99 = llvm.add %96, %98 : i64
    %100 = llvm.extractvalue %23[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %101 = llvm.mul %69, %100 : i64
    %102 = llvm.add %99, %101 : i64
    %103 = llvm.getelementptr %91[%102] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %104 = llvm.load %103 : !llvm.ptr -> f32
    %105 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %106 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %107 = llvm.getelementptr %105[%106] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %108 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %109 = llvm.mul %39, %108 : i64
    %110 = llvm.extractvalue %11[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %111 = llvm.mul %44, %110 : i64
    %112 = llvm.add %109, %111 : i64
    %113 = llvm.extractvalue %11[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %114 = llvm.mul %49, %113 : i64
    %115 = llvm.add %112, %114 : i64
    %116 = llvm.extractvalue %11[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %117 = llvm.mul %54, %116 : i64
    %118 = llvm.add %115, %117 : i64
    %119 = llvm.getelementptr %107[%118] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %120 = llvm.load %119 : !llvm.ptr -> f32
    %121 = llvm.fmul %88, %104  : f32
    %122 = llvm.fadd %120, %121  : f32
    %123 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %124 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %125 = llvm.getelementptr %123[%124] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %126 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %127 = llvm.mul %39, %126 : i64
    %128 = llvm.extractvalue %11[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %129 = llvm.mul %44, %128 : i64
    %130 = llvm.add %127, %129 : i64
    %131 = llvm.extractvalue %11[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %132 = llvm.mul %49, %131 : i64
    %133 = llvm.add %130, %132 : i64
    %134 = llvm.extractvalue %11[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %135 = llvm.mul %54, %134 : i64
    %136 = llvm.add %133, %135 : i64
    %137 = llvm.getelementptr %125[%136] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %122, %137 : f32, !llvm.ptr
    %138 = llvm.add %69, %68 : i64
    llvm.br ^bb13(%138 : i64)
  ^bb15:  // pred: ^bb13
    %139 = llvm.add %64, %63 : i64
    llvm.br ^bb11(%139 : i64)
  ^bb16:  // pred: ^bb11
    %140 = llvm.add %59, %58 : i64
    llvm.br ^bb9(%140 : i64)
  ^bb17:  // pred: ^bb9
    %141 = llvm.add %54, %53 : i64
    llvm.br ^bb7(%141 : i64)
  ^bb18:  // pred: ^bb7
    %142 = llvm.add %49, %48 : i64
    llvm.br ^bb5(%142 : i64)
  ^bb19:  // pred: ^bb5
    %143 = llvm.add %44, %43 : i64
    llvm.br ^bb3(%143 : i64)
  ^bb20:  // pred: ^bb3
    %144 = llvm.add %39, %38 : i64
    llvm.br ^bb1(%144 : i64)
  ^bb21:  // pred: ^bb1
    llvm.return %11 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant(1.474560e+05 : f64) : f64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(8 : index) : i64
    %3 = llvm.mlir.constant(12 : index) : i64
    %4 = llvm.mlir.constant(12 : index) : i64
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.mlir.constant(144 : index) : i64
    %7 = llvm.mlir.constant(1152 : index) : i64
    %8 = llvm.mlir.constant(1152 : index) : i64
    %9 = llvm.mlir.zero : !llvm.ptr
    %10 = llvm.getelementptr %9[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %11 = llvm.ptrtoint %10 : !llvm.ptr to i64
    %12 = llvm.mlir.constant(64 : index) : i64
    %13 = llvm.add %11, %12 : i64
    %14 = llvm.call @malloc(%13) : (i64) -> !llvm.ptr
    %15 = llvm.ptrtoint %14 : !llvm.ptr to i64
    %16 = llvm.mlir.constant(1 : index) : i64
    %17 = llvm.sub %12, %16 : i64
    %18 = llvm.add %15, %17 : i64
    %19 = llvm.urem %18, %12  : i64
    %20 = llvm.sub %18, %19 : i64
    %21 = llvm.inttoptr %20 : i64 to !llvm.ptr
    %22 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %23 = llvm.insertvalue %14, %22[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %24 = llvm.insertvalue %21, %23[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %25 = llvm.mlir.constant(0 : index) : i64
    %26 = llvm.insertvalue %25, %24[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %27 = llvm.insertvalue %1, %26[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %28 = llvm.insertvalue %2, %27[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %29 = llvm.insertvalue %3, %28[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %30 = llvm.insertvalue %4, %29[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %31 = llvm.insertvalue %7, %30[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %32 = llvm.insertvalue %6, %31[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %33 = llvm.insertvalue %4, %32[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %34 = llvm.insertvalue %5, %33[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %35 = llvm.mlir.constant(0 : index) : i64
    %36 = llvm.mlir.constant(1 : index) : i64
    %37 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%35 : i64)
  ^bb1(%38: i64):  // 2 preds: ^bb0, ^bb11
    %39 = llvm.icmp "slt" %38, %36 : i64
    llvm.cond_br %39, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    %40 = llvm.mlir.constant(0 : index) : i64
    %41 = llvm.mlir.constant(8 : index) : i64
    %42 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb3(%40 : i64)
  ^bb3(%43: i64):  // 2 preds: ^bb2, ^bb10
    %44 = llvm.icmp "slt" %43, %41 : i64
    llvm.cond_br %44, ^bb4, ^bb11
  ^bb4:  // pred: ^bb3
    %45 = llvm.mlir.constant(0 : index) : i64
    %46 = llvm.mlir.constant(12 : index) : i64
    %47 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb5(%45 : i64)
  ^bb5(%48: i64):  // 2 preds: ^bb4, ^bb9
    %49 = llvm.icmp "slt" %48, %46 : i64
    llvm.cond_br %49, ^bb6, ^bb10
  ^bb6:  // pred: ^bb5
    %50 = llvm.mlir.constant(0 : index) : i64
    %51 = llvm.mlir.constant(12 : index) : i64
    %52 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb7(%50 : i64)
  ^bb7(%53: i64):  // 2 preds: ^bb6, ^bb8
    %54 = llvm.icmp "slt" %53, %51 : i64
    llvm.cond_br %54, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %55 = llvm.add %53, %48 : i64
    %56 = llvm.add %55, %43 : i64
    %57 = llvm.add %56, %38 : i64
    %58 = llvm.uitofp %57 : i64 to f32
    %59 = llvm.extractvalue %34[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %60 = llvm.mlir.constant(1152 : index) : i64
    %61 = llvm.mul %38, %60 : i64
    %62 = llvm.mlir.constant(144 : index) : i64
    %63 = llvm.mul %43, %62 : i64
    %64 = llvm.add %61, %63 : i64
    %65 = llvm.mlir.constant(12 : index) : i64
    %66 = llvm.mul %48, %65 : i64
    %67 = llvm.add %64, %66 : i64
    %68 = llvm.add %67, %53 : i64
    %69 = llvm.getelementptr %59[%68] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %58, %69 : f32, !llvm.ptr
    %70 = llvm.add %53, %52 : i64
    llvm.br ^bb7(%70 : i64)
  ^bb9:  // pred: ^bb7
    %71 = llvm.add %48, %47 : i64
    llvm.br ^bb5(%71 : i64)
  ^bb10:  // pred: ^bb5
    %72 = llvm.add %43, %42 : i64
    llvm.br ^bb3(%72 : i64)
  ^bb11:  // pred: ^bb3
    %73 = llvm.add %38, %37 : i64
    llvm.br ^bb1(%73 : i64)
  ^bb12:  // pred: ^bb1
    %74 = llvm.mlir.constant(16 : index) : i64
    %75 = llvm.mlir.constant(8 : index) : i64
    %76 = llvm.mlir.constant(3 : index) : i64
    %77 = llvm.mlir.constant(3 : index) : i64
    %78 = llvm.mlir.constant(1 : index) : i64
    %79 = llvm.mlir.constant(9 : index) : i64
    %80 = llvm.mlir.constant(72 : index) : i64
    %81 = llvm.mlir.constant(1152 : index) : i64
    %82 = llvm.mlir.zero : !llvm.ptr
    %83 = llvm.getelementptr %82[%81] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %84 = llvm.ptrtoint %83 : !llvm.ptr to i64
    %85 = llvm.mlir.constant(64 : index) : i64
    %86 = llvm.add %84, %85 : i64
    %87 = llvm.call @malloc(%86) : (i64) -> !llvm.ptr
    %88 = llvm.ptrtoint %87 : !llvm.ptr to i64
    %89 = llvm.mlir.constant(1 : index) : i64
    %90 = llvm.sub %85, %89 : i64
    %91 = llvm.add %88, %90 : i64
    %92 = llvm.urem %91, %85  : i64
    %93 = llvm.sub %91, %92 : i64
    %94 = llvm.inttoptr %93 : i64 to !llvm.ptr
    %95 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %96 = llvm.insertvalue %87, %95[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %97 = llvm.insertvalue %94, %96[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %98 = llvm.mlir.constant(0 : index) : i64
    %99 = llvm.insertvalue %98, %97[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %100 = llvm.insertvalue %74, %99[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %101 = llvm.insertvalue %75, %100[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %102 = llvm.insertvalue %76, %101[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %103 = llvm.insertvalue %77, %102[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %104 = llvm.insertvalue %80, %103[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %105 = llvm.insertvalue %79, %104[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %106 = llvm.insertvalue %77, %105[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %107 = llvm.insertvalue %78, %106[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %108 = llvm.mlir.constant(0 : index) : i64
    %109 = llvm.mlir.constant(16 : index) : i64
    %110 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb13(%108 : i64)
  ^bb13(%111: i64):  // 2 preds: ^bb12, ^bb23
    %112 = llvm.icmp "slt" %111, %109 : i64
    llvm.cond_br %112, ^bb14, ^bb24
  ^bb14:  // pred: ^bb13
    %113 = llvm.mlir.constant(0 : index) : i64
    %114 = llvm.mlir.constant(8 : index) : i64
    %115 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb15(%113 : i64)
  ^bb15(%116: i64):  // 2 preds: ^bb14, ^bb22
    %117 = llvm.icmp "slt" %116, %114 : i64
    llvm.cond_br %117, ^bb16, ^bb23
  ^bb16:  // pred: ^bb15
    %118 = llvm.mlir.constant(0 : index) : i64
    %119 = llvm.mlir.constant(3 : index) : i64
    %120 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb17(%118 : i64)
  ^bb17(%121: i64):  // 2 preds: ^bb16, ^bb21
    %122 = llvm.icmp "slt" %121, %119 : i64
    llvm.cond_br %122, ^bb18, ^bb22
  ^bb18:  // pred: ^bb17
    %123 = llvm.mlir.constant(0 : index) : i64
    %124 = llvm.mlir.constant(3 : index) : i64
    %125 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb19(%123 : i64)
  ^bb19(%126: i64):  // 2 preds: ^bb18, ^bb20
    %127 = llvm.icmp "slt" %126, %124 : i64
    llvm.cond_br %127, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    %128 = llvm.add %126, %121 : i64
    %129 = llvm.add %128, %116 : i64
    %130 = llvm.add %129, %111 : i64
    %131 = llvm.uitofp %130 : i64 to f32
    %132 = llvm.extractvalue %107[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %133 = llvm.mlir.constant(72 : index) : i64
    %134 = llvm.mul %111, %133 : i64
    %135 = llvm.mlir.constant(9 : index) : i64
    %136 = llvm.mul %116, %135 : i64
    %137 = llvm.add %134, %136 : i64
    %138 = llvm.mlir.constant(3 : index) : i64
    %139 = llvm.mul %121, %138 : i64
    %140 = llvm.add %137, %139 : i64
    %141 = llvm.add %140, %126 : i64
    %142 = llvm.getelementptr %132[%141] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %131, %142 : f32, !llvm.ptr
    %143 = llvm.add %126, %125 : i64
    llvm.br ^bb19(%143 : i64)
  ^bb21:  // pred: ^bb19
    %144 = llvm.add %121, %120 : i64
    llvm.br ^bb17(%144 : i64)
  ^bb22:  // pred: ^bb17
    %145 = llvm.add %116, %115 : i64
    llvm.br ^bb15(%145 : i64)
  ^bb23:  // pred: ^bb15
    %146 = llvm.add %111, %110 : i64
    llvm.br ^bb13(%146 : i64)
  ^bb24:  // pred: ^bb13
    %147 = llvm.mlir.constant(1 : index) : i64
    %148 = llvm.mlir.constant(16 : index) : i64
    %149 = llvm.mlir.constant(8 : index) : i64
    %150 = llvm.mlir.constant(8 : index) : i64
    %151 = llvm.mlir.constant(1 : index) : i64
    %152 = llvm.mlir.constant(64 : index) : i64
    %153 = llvm.mlir.constant(1024 : index) : i64
    %154 = llvm.mlir.constant(1024 : index) : i64
    %155 = llvm.mlir.zero : !llvm.ptr
    %156 = llvm.getelementptr %155[%154] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %157 = llvm.ptrtoint %156 : !llvm.ptr to i64
    %158 = llvm.mlir.constant(64 : index) : i64
    %159 = llvm.add %157, %158 : i64
    %160 = llvm.call @malloc(%159) : (i64) -> !llvm.ptr
    %161 = llvm.ptrtoint %160 : !llvm.ptr to i64
    %162 = llvm.mlir.constant(1 : index) : i64
    %163 = llvm.sub %158, %162 : i64
    %164 = llvm.add %161, %163 : i64
    %165 = llvm.urem %164, %158  : i64
    %166 = llvm.sub %164, %165 : i64
    %167 = llvm.inttoptr %166 : i64 to !llvm.ptr
    %168 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %169 = llvm.insertvalue %160, %168[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %170 = llvm.insertvalue %167, %169[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %171 = llvm.mlir.constant(0 : index) : i64
    %172 = llvm.insertvalue %171, %170[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %173 = llvm.insertvalue %147, %172[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %174 = llvm.insertvalue %148, %173[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %175 = llvm.insertvalue %149, %174[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %176 = llvm.insertvalue %150, %175[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %177 = llvm.insertvalue %153, %176[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %178 = llvm.insertvalue %152, %177[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %179 = llvm.insertvalue %150, %178[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %180 = llvm.insertvalue %151, %179[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %181 = llvm.mlir.constant(0 : index) : i64
    %182 = llvm.mlir.constant(1 : index) : i64
    %183 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb25(%181 : i64)
  ^bb25(%184: i64):  // 2 preds: ^bb24, ^bb35
    %185 = llvm.icmp "slt" %184, %182 : i64
    llvm.cond_br %185, ^bb26, ^bb36
  ^bb26:  // pred: ^bb25
    %186 = llvm.mlir.constant(0 : index) : i64
    %187 = llvm.mlir.constant(16 : index) : i64
    %188 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb27(%186 : i64)
  ^bb27(%189: i64):  // 2 preds: ^bb26, ^bb34
    %190 = llvm.icmp "slt" %189, %187 : i64
    llvm.cond_br %190, ^bb28, ^bb35
  ^bb28:  // pred: ^bb27
    %191 = llvm.mlir.constant(0 : index) : i64
    %192 = llvm.mlir.constant(8 : index) : i64
    %193 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb29(%191 : i64)
  ^bb29(%194: i64):  // 2 preds: ^bb28, ^bb33
    %195 = llvm.icmp "slt" %194, %192 : i64
    llvm.cond_br %195, ^bb30, ^bb34
  ^bb30:  // pred: ^bb29
    %196 = llvm.mlir.constant(0 : index) : i64
    %197 = llvm.mlir.constant(8 : index) : i64
    %198 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb31(%196 : i64)
  ^bb31(%199: i64):  // 2 preds: ^bb30, ^bb32
    %200 = llvm.icmp "slt" %199, %197 : i64
    llvm.cond_br %200, ^bb32, ^bb33
  ^bb32:  // pred: ^bb31
    %201 = llvm.add %199, %194 : i64
    %202 = llvm.add %201, %189 : i64
    %203 = llvm.add %202, %184 : i64
    %204 = llvm.uitofp %203 : i64 to f32
    %205 = llvm.extractvalue %180[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %206 = llvm.mlir.constant(1024 : index) : i64
    %207 = llvm.mul %184, %206 : i64
    %208 = llvm.mlir.constant(64 : index) : i64
    %209 = llvm.mul %189, %208 : i64
    %210 = llvm.add %207, %209 : i64
    %211 = llvm.mlir.constant(8 : index) : i64
    %212 = llvm.mul %194, %211 : i64
    %213 = llvm.add %210, %212 : i64
    %214 = llvm.add %213, %199 : i64
    %215 = llvm.getelementptr %205[%214] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %204, %215 : f32, !llvm.ptr
    %216 = llvm.add %199, %198 : i64
    llvm.br ^bb31(%216 : i64)
  ^bb33:  // pred: ^bb31
    %217 = llvm.add %194, %193 : i64
    llvm.br ^bb29(%217 : i64)
  ^bb34:  // pred: ^bb29
    %218 = llvm.add %189, %188 : i64
    llvm.br ^bb27(%218 : i64)
  ^bb35:  // pred: ^bb27
    %219 = llvm.add %184, %183 : i64
    llvm.br ^bb25(%219 : i64)
  ^bb36:  // pred: ^bb25
    %220 = llvm.call @rtclock() : () -> f64
    %221 = llvm.extractvalue %34[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %222 = llvm.extractvalue %34[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %223 = llvm.extractvalue %34[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %224 = llvm.extractvalue %34[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %225 = llvm.extractvalue %34[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %226 = llvm.extractvalue %34[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %227 = llvm.extractvalue %34[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %228 = llvm.extractvalue %34[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %229 = llvm.extractvalue %34[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %230 = llvm.extractvalue %34[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %231 = llvm.extractvalue %34[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %232 = llvm.extractvalue %107[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %233 = llvm.extractvalue %107[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %234 = llvm.extractvalue %107[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %235 = llvm.extractvalue %107[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %236 = llvm.extractvalue %107[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %237 = llvm.extractvalue %107[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %238 = llvm.extractvalue %107[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %239 = llvm.extractvalue %107[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %240 = llvm.extractvalue %107[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %241 = llvm.extractvalue %107[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %242 = llvm.extractvalue %107[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %243 = llvm.extractvalue %180[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %244 = llvm.extractvalue %180[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %245 = llvm.extractvalue %180[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %246 = llvm.extractvalue %180[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %247 = llvm.extractvalue %180[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %248 = llvm.extractvalue %180[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %249 = llvm.extractvalue %180[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %250 = llvm.extractvalue %180[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %251 = llvm.extractvalue %180[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %252 = llvm.extractvalue %180[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %253 = llvm.extractvalue %180[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %254 = llvm.call @conv_2d_nchw_fchw(%221, %222, %223, %224, %225, %226, %227, %228, %229, %230, %231, %232, %233, %234, %235, %236, %237, %238, %239, %240, %241, %242, %243, %244, %245, %246, %247, %248, %249, %250, %251, %252, %253) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %255 = llvm.call @rtclock() : () -> f64
    %256 = llvm.fsub %255, %220  : f64
    %257 = llvm.mlir.constant(1 : index) : i64
    %258 = llvm.alloca %257 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %254, %258 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %259 = llvm.mlir.constant(4 : index) : i64
    %260 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %261 = llvm.insertvalue %259, %260[0] : !llvm.struct<(i64, ptr)> 
    %262 = llvm.insertvalue %258, %261[1] : !llvm.struct<(i64, ptr)> 
    %263 = llvm.extractvalue %262[0] : !llvm.struct<(i64, ptr)> 
    %264 = llvm.extractvalue %262[1] : !llvm.struct<(i64, ptr)> 
    llvm.call @printChecksumF32(%263, %264) : (i64, !llvm.ptr) -> ()
    %265 = llvm.fdiv %0, %256  : f64
    llvm.call @printFlops(%265) : (f64) -> ()
    llvm.return
  }
  llvm.func private @printChecksumF32(%arg0: i64, %arg1: !llvm.ptr) attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i64, ptr)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i64, ptr)> 
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.alloca %3 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %2, %4 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    llvm.call @_mlir_ciface_printChecksumF32(%4) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_printChecksumF32(!llvm.ptr) attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func @printFlops(f64) attributes {sym_visibility = "private"}
  llvm.func @rtclock() -> f64 attributes {sym_visibility = "private"}
}


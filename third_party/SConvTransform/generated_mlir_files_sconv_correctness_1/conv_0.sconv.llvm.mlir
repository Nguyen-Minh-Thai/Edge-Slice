module {
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
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
    %36 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %37 = llvm.mlir.constant(64 : i64) : i64
    %38 = llvm.mlir.constant(72 : i64) : i64
    %39 = llvm.mlir.constant(8 : i64) : i64
    %40 = llvm.mlir.constant(16 : i64) : i64
    %41 = llvm.mlir.constant(16 : index) : i64
    %42 = llvm.mlir.constant(64 : index) : i64
    %43 = llvm.mlir.constant(8 : index) : i64
    %44 = llvm.mlir.constant(0 : index) : i64
    %45 = llvm.mlir.constant(1 : index) : i64
    %46 = llvm.mlir.constant(8 : index) : i64
    %47 = llvm.mlir.constant(12 : index) : i64
    %48 = llvm.mlir.constant(12 : index) : i64
    %49 = llvm.mlir.constant(1 : index) : i64
    %50 = llvm.mlir.constant(144 : index) : i64
    %51 = llvm.mlir.constant(1152 : index) : i64
    %52 = llvm.mlir.constant(1152 : index) : i64
    %53 = llvm.mlir.zero : !llvm.ptr
    %54 = llvm.getelementptr %53[%52] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %55 = llvm.ptrtoint %54 : !llvm.ptr to i64
    %56 = llvm.mlir.constant(64 : index) : i64
    %57 = llvm.add %55, %56 : i64
    %58 = llvm.call @malloc(%57) : (i64) -> !llvm.ptr
    %59 = llvm.ptrtoint %58 : !llvm.ptr to i64
    %60 = llvm.mlir.constant(1 : index) : i64
    %61 = llvm.sub %56, %60 : i64
    %62 = llvm.add %59, %61 : i64
    %63 = llvm.urem %62, %56  : i64
    %64 = llvm.sub %62, %63 : i64
    %65 = llvm.inttoptr %64 : i64 to !llvm.ptr
    %66 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %67 = llvm.insertvalue %58, %66[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %68 = llvm.insertvalue %65, %67[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %69 = llvm.mlir.constant(0 : index) : i64
    %70 = llvm.insertvalue %69, %68[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %71 = llvm.insertvalue %45, %70[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %72 = llvm.insertvalue %46, %71[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %73 = llvm.insertvalue %47, %72[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %74 = llvm.insertvalue %48, %73[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %75 = llvm.insertvalue %51, %74[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %76 = llvm.insertvalue %50, %75[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %77 = llvm.insertvalue %48, %76[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %78 = llvm.insertvalue %49, %77[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %79 = llvm.intr.stacksave : !llvm.ptr
    %80 = llvm.mlir.constant(4 : i64) : i64
    %81 = llvm.mlir.constant(1 : index) : i64
    %82 = llvm.alloca %81 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %35, %82 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %83 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %84 = llvm.insertvalue %80, %83[0] : !llvm.struct<(i64, ptr)> 
    %85 = llvm.insertvalue %82, %84[1] : !llvm.struct<(i64, ptr)> 
    %86 = llvm.mlir.constant(4 : i64) : i64
    %87 = llvm.mlir.constant(1 : index) : i64
    %88 = llvm.alloca %87 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %78, %88 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %89 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %90 = llvm.insertvalue %86, %89[0] : !llvm.struct<(i64, ptr)> 
    %91 = llvm.insertvalue %88, %90[1] : !llvm.struct<(i64, ptr)> 
    %92 = llvm.mlir.constant(1 : index) : i64
    %93 = llvm.alloca %92 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %85, %93 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %94 = llvm.alloca %92 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %91, %94 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %95 = llvm.mlir.zero : !llvm.ptr
    %96 = llvm.getelementptr %95[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %97 = llvm.ptrtoint %96 : !llvm.ptr to i64
    llvm.call @memrefCopy(%97, %93, %94) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %79 : !llvm.ptr
    %98 = llvm.mlir.constant(1 : index) : i64
    %99 = llvm.mlir.constant(16 : index) : i64
    %100 = llvm.mlir.constant(8 : index) : i64
    %101 = llvm.mlir.constant(8 : index) : i64
    %102 = llvm.mlir.constant(1 : index) : i64
    %103 = llvm.mlir.constant(64 : index) : i64
    %104 = llvm.mlir.constant(1024 : index) : i64
    %105 = llvm.mlir.constant(1024 : index) : i64
    %106 = llvm.mlir.zero : !llvm.ptr
    %107 = llvm.getelementptr %106[%105] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %108 = llvm.ptrtoint %107 : !llvm.ptr to i64
    %109 = llvm.mlir.constant(64 : index) : i64
    %110 = llvm.add %108, %109 : i64
    %111 = llvm.call @malloc(%110) : (i64) -> !llvm.ptr
    %112 = llvm.ptrtoint %111 : !llvm.ptr to i64
    %113 = llvm.mlir.constant(1 : index) : i64
    %114 = llvm.sub %109, %113 : i64
    %115 = llvm.add %112, %114 : i64
    %116 = llvm.urem %115, %109  : i64
    %117 = llvm.sub %115, %116 : i64
    %118 = llvm.inttoptr %117 : i64 to !llvm.ptr
    %119 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %120 = llvm.insertvalue %111, %119[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %121 = llvm.insertvalue %118, %120[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %122 = llvm.mlir.constant(0 : index) : i64
    %123 = llvm.insertvalue %122, %121[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %124 = llvm.insertvalue %98, %123[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %125 = llvm.insertvalue %99, %124[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %126 = llvm.insertvalue %100, %125[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %127 = llvm.insertvalue %101, %126[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %128 = llvm.insertvalue %104, %127[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %129 = llvm.insertvalue %103, %128[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %130 = llvm.insertvalue %101, %129[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %131 = llvm.insertvalue %102, %130[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %132 = llvm.intr.stacksave : !llvm.ptr
    %133 = llvm.mlir.constant(4 : i64) : i64
    %134 = llvm.mlir.constant(1 : index) : i64
    %135 = llvm.alloca %134 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %11, %135 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %136 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %137 = llvm.insertvalue %133, %136[0] : !llvm.struct<(i64, ptr)> 
    %138 = llvm.insertvalue %135, %137[1] : !llvm.struct<(i64, ptr)> 
    %139 = llvm.mlir.constant(4 : i64) : i64
    %140 = llvm.mlir.constant(1 : index) : i64
    %141 = llvm.alloca %140 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %131, %141 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %142 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %143 = llvm.insertvalue %139, %142[0] : !llvm.struct<(i64, ptr)> 
    %144 = llvm.insertvalue %141, %143[1] : !llvm.struct<(i64, ptr)> 
    %145 = llvm.mlir.constant(1 : index) : i64
    %146 = llvm.alloca %145 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %138, %146 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %147 = llvm.alloca %145 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %144, %147 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %148 = llvm.mlir.zero : !llvm.ptr
    %149 = llvm.getelementptr %148[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %150 = llvm.ptrtoint %149 : !llvm.ptr to i64
    llvm.call @memrefCopy(%150, %146, %147) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %132 : !llvm.ptr
    %151 = llvm.mlir.constant(2 : index) : i64
    %152 = llvm.mlir.constant(8 : index) : i64
    %153 = llvm.mlir.constant(3 : index) : i64
    %154 = llvm.mlir.constant(3 : index) : i64
    %155 = llvm.mlir.constant(8 : index) : i64
    %156 = llvm.mlir.constant(1 : index) : i64
    %157 = llvm.mlir.constant(24 : index) : i64
    %158 = llvm.mlir.constant(72 : index) : i64
    %159 = llvm.mlir.constant(576 : index) : i64
    %160 = llvm.mlir.constant(1152 : index) : i64
    %161 = llvm.mlir.zero : !llvm.ptr
    %162 = llvm.getelementptr %161[%160] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %163 = llvm.ptrtoint %162 : !llvm.ptr to i64
    %164 = llvm.mlir.constant(64 : index) : i64
    %165 = llvm.add %163, %164 : i64
    %166 = llvm.call @malloc(%165) : (i64) -> !llvm.ptr
    %167 = llvm.ptrtoint %166 : !llvm.ptr to i64
    %168 = llvm.mlir.constant(1 : index) : i64
    %169 = llvm.sub %164, %168 : i64
    %170 = llvm.add %167, %169 : i64
    %171 = llvm.urem %170, %164  : i64
    %172 = llvm.sub %170, %171 : i64
    %173 = llvm.inttoptr %172 : i64 to !llvm.ptr
    %174 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)>
    %175 = llvm.insertvalue %166, %174[0] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %176 = llvm.insertvalue %173, %175[1] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %177 = llvm.mlir.constant(0 : index) : i64
    %178 = llvm.insertvalue %177, %176[2] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %179 = llvm.insertvalue %151, %178[3, 0] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %180 = llvm.insertvalue %152, %179[3, 1] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %181 = llvm.insertvalue %153, %180[3, 2] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %182 = llvm.insertvalue %154, %181[3, 3] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %183 = llvm.insertvalue %155, %182[3, 4] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %184 = llvm.insertvalue %159, %183[4, 0] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %185 = llvm.insertvalue %158, %184[4, 1] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %186 = llvm.insertvalue %157, %185[4, 2] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %187 = llvm.insertvalue %155, %186[4, 3] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %188 = llvm.insertvalue %156, %187[4, 4] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %189 = llvm.mlir.constant(0 : index) : i64
    %190 = llvm.mlir.constant(2 : index) : i64
    %191 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%189 : i64)
  ^bb1(%192: i64):  // 2 preds: ^bb0, ^bb14
    %193 = llvm.icmp "slt" %192, %190 : i64
    llvm.cond_br %193, ^bb2, ^bb15
  ^bb2:  // pred: ^bb1
    %194 = llvm.mlir.constant(0 : index) : i64
    %195 = llvm.mlir.constant(8 : index) : i64
    %196 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb3(%194 : i64)
  ^bb3(%197: i64):  // 2 preds: ^bb2, ^bb13
    %198 = llvm.icmp "slt" %197, %195 : i64
    llvm.cond_br %198, ^bb4, ^bb14
  ^bb4:  // pred: ^bb3
    %199 = llvm.mlir.constant(0 : index) : i64
    %200 = llvm.mlir.constant(3 : index) : i64
    %201 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb5(%199 : i64)
  ^bb5(%202: i64):  // 2 preds: ^bb4, ^bb12
    %203 = llvm.icmp "slt" %202, %200 : i64
    llvm.cond_br %203, ^bb6, ^bb13
  ^bb6:  // pred: ^bb5
    %204 = llvm.mlir.constant(0 : index) : i64
    %205 = llvm.mlir.constant(3 : index) : i64
    %206 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb7(%204 : i64)
  ^bb7(%207: i64):  // 2 preds: ^bb6, ^bb11
    %208 = llvm.icmp "slt" %207, %205 : i64
    llvm.cond_br %208, ^bb8, ^bb12
  ^bb8:  // pred: ^bb7
    %209 = llvm.mlir.constant(0 : index) : i64
    %210 = llvm.mlir.constant(8 : index) : i64
    %211 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb9(%209 : i64)
  ^bb9(%212: i64):  // 2 preds: ^bb8, ^bb10
    %213 = llvm.icmp "slt" %212, %210 : i64
    llvm.cond_br %213, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %214 = llvm.extractvalue %23[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %215 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %216 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %217 = llvm.insertvalue %214, %216[0] : !llvm.struct<(ptr, ptr, i64)> 
    %218 = llvm.insertvalue %215, %217[1] : !llvm.struct<(ptr, ptr, i64)> 
    %219 = llvm.mlir.constant(0 : index) : i64
    %220 = llvm.insertvalue %219, %218[2] : !llvm.struct<(ptr, ptr, i64)> 
    %221 = llvm.extractvalue %23[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %222 = llvm.extractvalue %23[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %223 = llvm.extractvalue %23[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %224 = llvm.extractvalue %23[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %225 = llvm.extractvalue %23[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %226 = llvm.extractvalue %23[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %227 = llvm.extractvalue %23[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %228 = llvm.extractvalue %23[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %229 = llvm.extractvalue %23[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %230 = llvm.mlir.constant(8 : index) : i64
    %231 = llvm.mul %192, %230 : i64
    %232 = llvm.add %231, %212 : i64
    %233 = llvm.mul %232, %226 : i64
    %234 = llvm.mul %197, %227 : i64
    %235 = llvm.mul %202, %228 : i64
    %236 = llvm.add %234, %235 : i64
    %237 = llvm.mul %207, %229 : i64
    %238 = llvm.add %236, %237 : i64
    %239 = llvm.add %238, %221 : i64
    %240 = llvm.add %233, %239 : i64
    %241 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %242 = llvm.extractvalue %220[0] : !llvm.struct<(ptr, ptr, i64)> 
    %243 = llvm.extractvalue %220[1] : !llvm.struct<(ptr, ptr, i64)> 
    %244 = llvm.insertvalue %242, %241[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %245 = llvm.insertvalue %243, %244[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %246 = llvm.insertvalue %240, %245[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %247 = llvm.mlir.constant(1 : index) : i64
    %248 = llvm.insertvalue %247, %246[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %249 = llvm.insertvalue %226, %248[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %250 = llvm.mlir.constant(1 : index) : i64
    %251 = llvm.insertvalue %250, %249[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %252 = llvm.insertvalue %227, %251[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %253 = llvm.mlir.constant(1 : index) : i64
    %254 = llvm.insertvalue %253, %252[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %255 = llvm.insertvalue %228, %254[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %256 = llvm.mlir.constant(1 : index) : i64
    %257 = llvm.insertvalue %256, %255[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %258 = llvm.insertvalue %229, %257[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %259 = llvm.extractvalue %258[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %260 = llvm.extractvalue %258[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %261 = llvm.getelementptr %259[%260] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %262 = llvm.extractvalue %258[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %263 = llvm.mul %44, %262 : i64
    %264 = llvm.extractvalue %258[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %265 = llvm.mul %44, %264 : i64
    %266 = llvm.add %263, %265 : i64
    %267 = llvm.extractvalue %258[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %268 = llvm.mul %44, %267 : i64
    %269 = llvm.add %266, %268 : i64
    %270 = llvm.extractvalue %258[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %271 = llvm.mul %44, %270 : i64
    %272 = llvm.add %269, %271 : i64
    %273 = llvm.getelementptr %261[%272] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %274 = llvm.load %273 : !llvm.ptr -> f32
    %275 = llvm.extractvalue %188[1] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %276 = llvm.mlir.constant(576 : index) : i64
    %277 = llvm.mul %192, %276 : i64
    %278 = llvm.mlir.constant(72 : index) : i64
    %279 = llvm.mul %197, %278 : i64
    %280 = llvm.add %277, %279 : i64
    %281 = llvm.mlir.constant(24 : index) : i64
    %282 = llvm.mul %202, %281 : i64
    %283 = llvm.add %280, %282 : i64
    %284 = llvm.mlir.constant(8 : index) : i64
    %285 = llvm.mul %207, %284 : i64
    %286 = llvm.add %283, %285 : i64
    %287 = llvm.add %286, %212 : i64
    %288 = llvm.getelementptr %275[%287] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %274, %288 : f32, !llvm.ptr
    %289 = llvm.add %212, %211 : i64
    llvm.br ^bb9(%289 : i64)
  ^bb11:  // pred: ^bb9
    %290 = llvm.add %207, %206 : i64
    llvm.br ^bb7(%290 : i64)
  ^bb12:  // pred: ^bb7
    %291 = llvm.add %202, %201 : i64
    llvm.br ^bb5(%291 : i64)
  ^bb13:  // pred: ^bb5
    %292 = llvm.add %197, %196 : i64
    llvm.br ^bb3(%292 : i64)
  ^bb14:  // pred: ^bb3
    %293 = llvm.add %192, %191 : i64
    llvm.br ^bb1(%293 : i64)
  ^bb15:  // pred: ^bb1
    llvm.br ^bb16(%44 : i64)
  ^bb16(%294: i64):  // 2 preds: ^bb15, ^bb35
    %295 = llvm.icmp "slt" %294, %42 : i64
    llvm.cond_br %295, ^bb17, ^bb36
  ^bb17:  // pred: ^bb16
    %296 = llvm.mlir.constant(1 : index) : i64
    %297 = llvm.mlir.constant(8 : index) : i64
    %298 = llvm.mlir.constant(3 : index) : i64
    %299 = llvm.mlir.constant(3 : index) : i64
    %300 = llvm.mlir.constant(16 : index) : i64
    %301 = llvm.mlir.constant(1 : index) : i64
    %302 = llvm.mlir.constant(48 : index) : i64
    %303 = llvm.mlir.constant(144 : index) : i64
    %304 = llvm.mlir.constant(1152 : index) : i64
    %305 = llvm.mlir.constant(1152 : index) : i64
    %306 = llvm.mlir.zero : !llvm.ptr
    %307 = llvm.getelementptr %306[%305] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %308 = llvm.ptrtoint %307 : !llvm.ptr to i64
    %309 = llvm.mlir.constant(64 : index) : i64
    %310 = llvm.add %308, %309 : i64
    %311 = llvm.call @malloc(%310) : (i64) -> !llvm.ptr
    %312 = llvm.ptrtoint %311 : !llvm.ptr to i64
    %313 = llvm.mlir.constant(1 : index) : i64
    %314 = llvm.sub %309, %313 : i64
    %315 = llvm.add %312, %314 : i64
    %316 = llvm.urem %315, %309  : i64
    %317 = llvm.sub %315, %316 : i64
    %318 = llvm.inttoptr %317 : i64 to !llvm.ptr
    %319 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)>
    %320 = llvm.insertvalue %311, %319[0] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %321 = llvm.insertvalue %318, %320[1] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %322 = llvm.mlir.constant(0 : index) : i64
    %323 = llvm.insertvalue %322, %321[2] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %324 = llvm.insertvalue %296, %323[3, 0] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %325 = llvm.insertvalue %297, %324[3, 1] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %326 = llvm.insertvalue %298, %325[3, 2] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %327 = llvm.insertvalue %299, %326[3, 3] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %328 = llvm.insertvalue %300, %327[3, 4] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %329 = llvm.insertvalue %304, %328[4, 0] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %330 = llvm.insertvalue %303, %329[4, 1] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %331 = llvm.insertvalue %302, %330[4, 2] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %332 = llvm.insertvalue %300, %331[4, 3] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %333 = llvm.insertvalue %301, %332[4, 4] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %334 = llvm.mlir.constant(0 : index) : i64
    %335 = llvm.mlir.constant(1 : index) : i64
    %336 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb18(%334 : i64)
  ^bb18(%337: i64):  // 2 preds: ^bb17, ^bb31
    %338 = llvm.icmp "slt" %337, %335 : i64
    llvm.cond_br %338, ^bb19, ^bb32
  ^bb19:  // pred: ^bb18
    %339 = llvm.mlir.constant(0 : index) : i64
    %340 = llvm.mlir.constant(8 : index) : i64
    %341 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb20(%339 : i64)
  ^bb20(%342: i64):  // 2 preds: ^bb19, ^bb30
    %343 = llvm.icmp "slt" %342, %340 : i64
    llvm.cond_br %343, ^bb21, ^bb31
  ^bb21:  // pred: ^bb20
    %344 = llvm.mlir.constant(0 : index) : i64
    %345 = llvm.mlir.constant(3 : index) : i64
    %346 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb22(%344 : i64)
  ^bb22(%347: i64):  // 2 preds: ^bb21, ^bb29
    %348 = llvm.icmp "slt" %347, %345 : i64
    llvm.cond_br %348, ^bb23, ^bb30
  ^bb23:  // pred: ^bb22
    %349 = llvm.mlir.constant(0 : index) : i64
    %350 = llvm.mlir.constant(3 : index) : i64
    %351 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb24(%349 : i64)
  ^bb24(%352: i64):  // 2 preds: ^bb23, ^bb28
    %353 = llvm.icmp "slt" %352, %350 : i64
    llvm.cond_br %353, ^bb25, ^bb29
  ^bb25:  // pred: ^bb24
    %354 = llvm.mlir.constant(0 : index) : i64
    %355 = llvm.mlir.constant(16 : index) : i64
    %356 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb26(%354 : i64)
  ^bb26(%357: i64):  // 2 preds: ^bb25, ^bb27
    %358 = llvm.icmp "slt" %357, %355 : i64
    llvm.cond_br %358, ^bb27, ^bb28
  ^bb27:  // pred: ^bb26
    %359 = llvm.mlir.constant(12 : index) : i64
    %360 = llvm.mul %347, %359 : i64
    %361 = llvm.add %360, %352 : i64
    %362 = llvm.add %361, %357 : i64
    %363 = llvm.add %362, %294 : i64
    %364 = llvm.mlir.constant(1152 : index) : i64
    %365 = llvm.mul %337, %364 : i64
    %366 = llvm.add %363, %365 : i64
    %367 = llvm.mlir.constant(144 : index) : i64
    %368 = llvm.mul %342, %367 : i64
    %369 = llvm.add %366, %368 : i64
    %370 = llvm.add %357, %294 : i64
    %371 = llvm.mlir.constant(8 : index) : i64
    %372 = llvm.mlir.constant(0 : index) : i64
    %373 = llvm.mlir.constant(-1 : index) : i64
    %374 = llvm.icmp "slt" %370, %372 : i64
    %375 = llvm.sub %373, %370 : i64
    %376 = llvm.select %374, %375, %370 : i1, i64
    %377 = llvm.sdiv %376, %371  : i64
    %378 = llvm.sub %373, %377 : i64
    %379 = llvm.select %374, %378, %377 : i1, i64
    %380 = llvm.mlir.constant(4 : index) : i64
    %381 = llvm.mul %379, %380 : i64
    %382 = llvm.add %369, %381 : i64
    %383 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %384 = llvm.extractvalue %78[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %385 = llvm.extractvalue %78[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %386 = llvm.insertvalue %384, %383[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %387 = llvm.insertvalue %385, %386[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %388 = llvm.insertvalue %382, %387[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %389 = llvm.mlir.constant(1 : index) : i64
    %390 = llvm.insertvalue %389, %388[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %391 = llvm.mlir.constant(1152 : index) : i64
    %392 = llvm.insertvalue %391, %390[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %393 = llvm.mlir.constant(1 : index) : i64
    %394 = llvm.insertvalue %393, %392[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %395 = llvm.mlir.constant(144 : index) : i64
    %396 = llvm.insertvalue %395, %394[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %397 = llvm.mlir.constant(1 : index) : i64
    %398 = llvm.insertvalue %397, %396[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %399 = llvm.mlir.constant(1 : index) : i64
    %400 = llvm.insertvalue %399, %398[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %401 = llvm.extractvalue %400[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %402 = llvm.extractvalue %400[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %403 = llvm.getelementptr %401[%402] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %404 = llvm.mlir.constant(1152 : index) : i64
    %405 = llvm.mul %44, %404 : i64
    %406 = llvm.mlir.constant(144 : index) : i64
    %407 = llvm.mul %44, %406 : i64
    %408 = llvm.add %405, %407 : i64
    %409 = llvm.add %408, %44 : i64
    %410 = llvm.getelementptr %403[%409] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %411 = llvm.load %410 : !llvm.ptr -> f32
    %412 = llvm.extractvalue %333[1] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %413 = llvm.mlir.constant(1152 : index) : i64
    %414 = llvm.mul %337, %413 : i64
    %415 = llvm.mlir.constant(144 : index) : i64
    %416 = llvm.mul %342, %415 : i64
    %417 = llvm.add %414, %416 : i64
    %418 = llvm.mlir.constant(48 : index) : i64
    %419 = llvm.mul %347, %418 : i64
    %420 = llvm.add %417, %419 : i64
    %421 = llvm.mlir.constant(16 : index) : i64
    %422 = llvm.mul %352, %421 : i64
    %423 = llvm.add %420, %422 : i64
    %424 = llvm.add %423, %357 : i64
    %425 = llvm.getelementptr %412[%424] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %411, %425 : f32, !llvm.ptr
    %426 = llvm.add %357, %356 : i64
    llvm.br ^bb26(%426 : i64)
  ^bb28:  // pred: ^bb26
    %427 = llvm.add %352, %351 : i64
    llvm.br ^bb24(%427 : i64)
  ^bb29:  // pred: ^bb24
    %428 = llvm.add %347, %346 : i64
    llvm.br ^bb22(%428 : i64)
  ^bb30:  // pred: ^bb22
    %429 = llvm.add %342, %341 : i64
    llvm.br ^bb20(%429 : i64)
  ^bb31:  // pred: ^bb20
    %430 = llvm.add %337, %336 : i64
    llvm.br ^bb18(%430 : i64)
  ^bb32:  // pred: ^bb18
    llvm.br ^bb33(%44 : i64)
  ^bb33(%431: i64):  // 2 preds: ^bb32, ^bb34
    %432 = llvm.icmp "slt" %431, %41 : i64
    llvm.cond_br %432, ^bb34, ^bb35
  ^bb34:  // pred: ^bb33
    %433 = llvm.mlir.constant(8 : index) : i64
    %434 = llvm.mlir.constant(0 : index) : i64
    %435 = llvm.mlir.constant(-1 : index) : i64
    %436 = llvm.icmp "slt" %431, %434 : i64
    %437 = llvm.sub %435, %431 : i64
    %438 = llvm.select %436, %437, %431 : i1, i64
    %439 = llvm.sdiv %438, %433  : i64
    %440 = llvm.sub %435, %439 : i64
    %441 = llvm.select %436, %440, %439 : i1, i64
    %442 = llvm.mlir.constant(576 : index) : i64
    %443 = llvm.mul %441, %442 : i64
    %444 = llvm.mlir.constant(64 : index) : i64
    %445 = llvm.mul %431, %444 : i64
    %446 = llvm.add %445, %294 : i64
    %447 = llvm.extractvalue %333[1] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %448 = llvm.ptrtoint %447 : !llvm.ptr to i64
    %449 = llvm.inttoptr %448 : i64 to !llvm.ptr
    %450 = llvm.extractvalue %188[1] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)> 
    %451 = llvm.ptrtoint %450 : !llvm.ptr to i64
    %452 = llvm.mlir.constant(4 : index) : i64
    %453 = llvm.mul %443, %452 : i64
    %454 = llvm.add %451, %453 : i64
    %455 = llvm.inttoptr %454 : i64 to !llvm.ptr
    %456 = llvm.extractvalue %131[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %457 = llvm.ptrtoint %456 : !llvm.ptr to i64
    %458 = llvm.mlir.constant(4 : index) : i64
    %459 = llvm.mul %446, %458 : i64
    %460 = llvm.add %457, %459 : i64
    %461 = llvm.inttoptr %460 : i64 to !llvm.ptr
    %462 = llvm.call @sgemm_blas_kernel(%40, %39, %38, %36, %449, %455, %461, %37) {schedule = "IS"} : (i64, i64, i64, f32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
    %463 = llvm.add %431, %43 : i64
    llvm.br ^bb33(%463 : i64)
  ^bb35:  // pred: ^bb33
    %464 = llvm.add %294, %41 : i64
    llvm.br ^bb16(%464 : i64)
  ^bb36:  // pred: ^bb16
    llvm.return %131 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
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
  llvm.func @sgemm_blas_kernel(i64, i64, i64, f32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
}


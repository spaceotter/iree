// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TODO(ataei): We are keeping this here for now to converge on the design of
// how K0, M0, N0 are suppose to choisen for compilation
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

class LinalgMatmulOpToLinalgMMT4dOpPattern
    : public OpRewritePattern<linalg::MatmulOp> {
 public:
  LinalgMatmulOpToLinalgMMT4dOpPattern(MLIRContext *context, int M0, int N0,
                                       int K0, PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::MatmulOp>(context, benefit),
        M0Size(M0),
        N0Size(N0),
        K0Size(K0) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    auto loc = matmulOp.getLoc();

    Value lhs = matmulOp.getInputOperand(0)->get();
    Value rhs = matmulOp.getInputOperand(1)->get();
    Value dst = matmulOp.getOutputOperand(0)->get();

    RankedTensorType lhsType = lhs.getType().dyn_cast<RankedTensorType>();
    RankedTensorType rhsType = rhs.getType().dyn_cast<RankedTensorType>();

    if (!lhsType || !rhsType || !lhsType.hasStaticShape() ||
        !rhsType.hasStaticShape()) {
      return failure();
    }

    int m = lhsType.getShape()[0];
    int n = rhsType.getShape()[1];
    int k = rhsType.getShape()[0];

    if (m % M0Size != 0 || n % N0Size != 0 || k % K0Size != 0) return failure();

    int m1 = m / M0Size;
    int n1 = n / N0Size;
    int k1 = k / K0Size;

    // Expands a 2d tensor operand to 4d given its target shape.
    auto expandTo4D = [&](Value operand,
                          ArrayRef<int64_t> targetShape) -> Value {
      auto operandType = operand.getType().cast<RankedTensorType>();
      auto targetType =
          RankedTensorType::get(targetShape, operandType.getElementType());
      SmallVector<ReassociationIndices> expandIndices = {{0, 1}, {2, 3}};
      Value reshapedOperand = rewriter.create<linalg::TensorExpandShapeOp>(
          loc, targetType, operand, expandIndices);
      return reshapedOperand;
    };

    auto lhs4D = expandTo4D(lhs, {m1, M0Size, k1, K0Size});
    auto rhs4D = expandTo4D(rhs, {k1, K0Size, n1, N0Size});
    auto dst4D = expandTo4D(dst, {m1, M0Size, n1, N0Size});

    auto transposeOperand = [&](Value operand,
                                ArrayRef<int64_t> indices) -> Value {
      RankedTensorType operandTensorType =
          operand.getType().cast<RankedTensorType>();
      auto nloops = indices.size();
      auto inputShape = operandTensorType.getShape();

      SmallVector<AffineExpr, 4> exprs = llvm::to_vector<4>(
          llvm::map_range(indices, [&](int64_t index) -> AffineExpr {
            return rewriter.getAffineDimExpr(index);
          }));

      SmallVector<int64_t> targetShape = llvm::to_vector<4>(llvm::map_range(
          indices,
          [&](int64_t index) -> int64_t { return inputShape[index]; }));

      Value outputTensor = rewriter.create<linalg::InitTensorOp>(
          loc, targetShape, operandTensorType.getElementType());

      SmallVector<StringRef> loopAttributeTypes(nloops, "parallel");

      SmallVector<AffineMap> indexingMaps = {
          inversePermutation(
              AffineMap::get(nloops, 0, exprs, rewriter.getContext())),
          AffineMap::getMultiDimIdentityMap(nloops, rewriter.getContext())};

      auto transposedOp = rewriter.create<linalg::GenericOp>(
          loc, outputTensor.getType(),
          /*inputs=*/operand, /*outputs=*/outputTensor, indexingMaps,
          loopAttributeTypes,
          [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
            nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
          });

      return transposedOp.getResult(0);
    };

    auto lhs4DT = transposeOperand(lhs4D, {0, 2, 1, 3});
    auto rhs4DT = transposeOperand(rhs4D, {2, 0, 3, 1});
    auto dst4DT = transposeOperand(dst4D, {0, 2, 1, 3});

    auto mmt4DResult = rewriter.create<linalg::Mmt4DOp>(
        loc, dst4DT.getType(), ValueRange{lhs4DT, rhs4DT}, ValueRange{dst4DT});

    auto mmt4dResultTransposed =
        transposeOperand(mmt4DResult.getResult(0), {0, 2, 1, 3});

    auto collapseTo2D = [&](Value operand,
                            ArrayRef<int64_t> targetShape) -> Value {
      auto operandType = operand.getType().cast<RankedTensorType>();
      auto targetType =
          RankedTensorType::get(targetShape, operandType.getElementType());
      SmallVector<ReassociationIndices> collapseIndices = {{0, 1}, {2, 3}};
      Value reshapedOperand = rewriter.create<linalg::TensorCollapseShapeOp>(
          loc, targetType, operand, collapseIndices);
      return reshapedOperand;
    };

    Value result = collapseTo2D(mmt4dResultTransposed, {m, n});

    rewriter.replaceOp(matmulOp, ArrayRef<Value>{result});

    return success();
  }

 private:
  int M0Size;
  int N0Size;
  int K0Size;
};

class ConvertLinalgMatmulOpToLinalgMMT4dPass
    : public ConvertMatmulToMMT4dBase<ConvertLinalgMatmulOpToLinalgMMT4dPass> {
 public:
  ConvertLinalgMatmulOpToLinalgMMT4dPass(int M0, int N0, int K0)
      : M0Size(M0), N0Size(K0), K0Size(N0) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    OwningRewritePatternList patterns(&getContext());
    patterns.insert<LinalgMatmulOpToLinalgMMT4dOpPattern>(context, M0Size,
                                                          N0Size, K0Size);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

 private:
  int M0Size;
  int N0Size;
  int K0Size;
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>>
createConvertLinalgMatmulOpToLinalgMMT4dPass(int M0, int N0, int K0) {
  return std::make_unique<ConvertLinalgMatmulOpToLinalgMMT4dPass>(M0, N0, K0);
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#map = affine_map<(d0) -> (d0)>
module  {
  func @forward(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> attributes {} {
    %c0 = arith.constant 0 : index
    %dim0 = tensor.dim %arg0, %c0 : tensor<?xf32>
    %0 = linalg.init_tensor [%dim0] : tensor<?xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) outs(%0 : tensor<?xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %5 = arith.subf %arg2, %arg3 : f32
      linalg.yield %5 : f32
    } -> tensor<?xf32>
    //%2 = tensor.cast %1 : tensor<?xf32> to tensor<?xf32>
    return %1 : tensor<?xf32>
  }
}

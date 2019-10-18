async function bufferTest(){
    tf.setBackend('webgpu');
    await tf.ready();
    let device = tf.backend().device;
    let a=tf.tensor1d([1,2,3,4], 'float32');
    let buffer=await tf.backend().getGPUBuffer(a.dataId);
    gpuReadBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      });
    
      const commandEncoder = device.createCommandEncoder({});
      commandEncoder.copyBufferToBuffer(
        buffer /* source buffer */,
        0 /* source offset */,
        gpuReadBuffer /* destination buffer */,
        0 /* destination offset */,
        16 /* size */
      );
      device.getQueue().submit([commandEncoder.finish()]);
      const copyArrayBuffer = await gpuReadBuffer.mapReadAsync();
      console.log(new Float32Array(copyArrayBuffer));
  }

function reduceDimension(arr) {
  return Array.prototype.concat.apply([], arr);
}
async function storageTest() {
  tf.setBackend('webgpu');
  await tf.ready();
  let device=tf.backend().device;
  const nn = navigator.ml.getNeuralNetworkContext();

  options={
    "backend": "WebML",
    "prefer": "sustained"
  };
  let model = await nn.createModel(options);
  let operandIndex = 0;

  let a=tf.tensor4d([1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0],[1,3,3,1],'float32');
  await a.data();
  let op1_value = [1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0];
  let op4_expect = [0.875, 0.875, 0.875, 0.875];

  let type3 = {type: nn.INT32};
  let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
  let type1_length = product(type1.dimensions);
  let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 1]};
  let type0_length = product(type0.dimensions);
  let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
  let type2_length = product(type2.dimensions);

  let op1 = operandIndex++;
  model.addOperand(type0);
  let op2 = operandIndex++;
  model.addOperand(type1);
  let op3 = operandIndex++;
  model.addOperand(type2);
  let pad0 = operandIndex++;
  model.addOperand(type3);
  let act = operandIndex++;
  model.addOperand(type3);
  let stride = operandIndex++;
  model.addOperand(type3);
  let op4 = operandIndex++;
  model.addOperand(type1);

  model.setOperandValue(op2, new Float32Array([0.25, 0.25, 0.25, 0.25]));
  model.setOperandValue(op3, new Float32Array([0]));
  model.setOperandValue(pad0, new Int32Array([0]));
  model.setOperandValue(act, new Int32Array([0]));
  model.setOperandValue(stride, new Int32Array([1]));
  model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);

  model.identifyInputsAndOutputs([op1], [op4]);
  await model.finish();

  let compilation = await model.createCompilation();
  compilation.setPreference(getPreferenceCode(options.prefer));
  await compilation.finish();

  let execution = await compilation.createExecution();

  //let op1_input = new Float32Array(op1_value);
  //execution.setInput(0, op1_input);
  let gpuWriteBuffer=await tf.backend().getGPUBuffer(a.dataId);
  let bufferSize=product(type0.dimensions)*4;
  let gpuReadBuffer = device.createBuffer({
    size: bufferSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const commandEncoder = device.createCommandEncoder({});
  commandEncoder.copyBufferToBuffer(
    gpuWriteBuffer /* source buffer */,
    0 /* source offset */,
    gpuReadBuffer /* destination buffer */,
    0 /* destination offset */,
    bufferSize /* size */
  );
  commandEncoder.shareBufferToWebml(gpuReadBuffer, 0);
  device.getQueue().submit([commandEncoder.finish()]);

  let op4_output = new Float32Array(type1_length);
  execution.setOutput(0, op4_output);

  await execution.startCompute();
  document.getElementById('op1').innerText =op4_expect;
  document.getElementById('op2').innerText =op4_output;}
async function storageFormat() {
  tf.setBackend('webgpu');
  await tf.ready();
  let device=tf.backend().device;
  const nn = navigator.ml.getNeuralNetworkContext();
  //let a=tf.truncatedNormal([2,2,2,2],1);
  let a=tf.tensor4d([1,2,3,4,5,6,7,8,9,0,11,12,13,14,15,16],[2,2,2,2],'float32');
  await a.data();
  options={
    "backend": "WebML",
    "prefer": "sustained"
  };
  let model = await nn.createModel(options);
  let operandIndex = 0;
  let op2_value = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];


  let type1 = {type: nn.INT32};
  let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2,2,2,2]};
  let type0_length = product(type0.dimensions);

  let op1 = operandIndex++;
  model.addOperand(type0);
  let op2 = operandIndex++;
  model.addOperand(type0);
  let act = operandIndex++;
  model.addOperand(type1);
  let op3 = operandIndex++;
  model.addOperand(type0);

  let op2_input = new Float32Array(op2_value);
  model.setOperandValue(op2, op2_input);

  model.setOperandValue(act, new Int32Array([0]));
  model.addOperation(nn.ADD, [op1, op2, act], [op3]);

  model.identifyInputsAndOutputs([op1], [op3]);
  await model.finish();

  let compilation = await model.createCompilation();
  compilation.setPreference(getPreferenceCode(options.prefer));
  await compilation.finish();

  let execution = await compilation.createExecution();

  //let op1_input = new Float32Array(op1_value);
  //execution.setInput(0, op1_input);
  let gpuWriteBuffer=await tf.backend().getGPUBuffer(a.dataId);
  let bufferSize=product(type0.dimensions)*4;
  let gpuReadBuffer = device.createBuffer({
    size: bufferSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const commandEncoder = device.createCommandEncoder({});
  commandEncoder.copyBufferToBuffer(
    gpuWriteBuffer /* source buffer */,
    0 /* source offset */,
    gpuReadBuffer /* destination buffer */,
    0 /* destination offset */,
    bufferSize /* size */
  );
  commandEncoder.shareBufferToWebml(gpuReadBuffer, 0);
  device.getQueue().submit([commandEncoder.finish()]);

  let op3_output = new Float32Array(type0_length);
  execution.setOutput(0, op3_output);

  await execution.startCompute();

  let arrayA=await a.array();
  let op3_expect = new Float32Array (arrayA.flatten().flatten().flatten());
  document.getElementById('op1').innerText =op3_expect;
  document.getElementById('op2').innerText =op3_output;}

storageFormat();
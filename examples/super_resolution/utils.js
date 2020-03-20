class Utils {
  constructor() {
    this.rawModel;
    this.model;
    this.initialized;
    this.updateProgress;
    this.modelFile;
    this.inputSize;
    this.outputSize;
    this.inputTensor;
    this.outputTensor;
    this.uData;
    this.vData;
    this.inCanvas = document.createElement('canvas');
    this.inCtx = this.inCanvas.getContext('2d');
    this.outCanvas = document.createElement('canvas');
    this.outCtx = this.outCanvas.getContext('2d');
    this.backend = '';
    this.prefer = '';
    this.initialized = false;
    this.loaded = false;
    this.resolveGetRequiredOps = null;
    this.outstandingRequest = null;
  }

  async loadModel(newModel) {
    if (this.loaded && this.modelFile === newModel.modelFile) {
      return 'LOADED';
    }
    // reset all states
    this.loaded = this.initialized = false;
    this.backend = this.prefer = '';

    this.modelFile = newModel.modelFile;
    this.inputSize = newModel.inputSize;
    this.outputSize = newModel.outputSize;
    this.inputTensor = new Float32Array(this.product(this.inputSize));
    this.outputTensor = new Float32Array(this.product(this.outputSize));
    this.uData = new Float32Array(this.product(this.inputSize));
    this.vData = new Float32Array(this.product(this.inputSize));
    this.rawModel = null;
    this.inCanvas.width = this.inputSize[3];
    this.inCanvas.height = this.inputSize[2];
    this.outCanvas.width = this.outputSize[3];
    this.outCanvas.height = this.outputSize[2];

    let result = await this.loadModelFile(this.modelFile);

    switch (this.modelFile.split('.').pop()) {
      case 'tflite':
        let flatBuffer = new flatbuffers.ByteBuffer(result);
        this.rawModel = tflite.Model.getRootAsModel(flatBuffer);
        this.rawModel._rawFormat = 'TFLITE';
        printTfLiteModel(this.rawModel);
        break;
      case 'onnx':
        let err = onnx.ModelProto.verify(result);
        if (err) {
          throw new Error(`Invalid model ${err}`);
        }
        this.rawModel = onnx.ModelProto.decode(result);
        this.rawModel._rawFormat = 'ONNX';
        printOnnxModel(this.rawModel);
        break;
      default:
        throw new Error('Unrecognized model format');
    }

    this.loaded = true;
    return 'SUCCESS';
  }

  async init(backend, prefer) {
    if (!this.loaded) {
      return 'NOT_LOADED';
    }
    if (this.initialized && backend === this.backend && prefer === this.prefer) {
      return 'INITIALIZED';
    }
    this.initialized = false;
    this.backend = backend;
    this.prefer = prefer;
    let kwargs = {
      rawModel: this.rawModel,
      backend: backend,
      prefer: prefer
    };
    switch (this.rawModel._rawFormat) {
      case 'TFLITE':
        this.model = new TFliteModelImporter(kwargs);
        break;
      case 'ONNX':
        this.model = new OnnxModelImporter(kwargs);
        break;
    }
    let result = await this.model.createCompiledModel();
    console.log(`compilation result: ${result}`);
    let start = performance.now();
    result = await this.model.compute([this.inputTensor], [this.outputTensor]);
    let elapsed = performance.now() - start;
    console.log(`warmup time: ${elapsed.toFixed(2)} ms`);
    this.initialized = true;

    if (this.resolveGetRequiredOps) {
      this.resolveGetRequiredOps(this.model.getRequiredOps());
    }

    return 'SUCCESS';
  }

  async getRequiredOps() {
    if (!this.initialized) {
      return new Promise(resolve => this.resolveGetRequiredOps = resolve);
    } else {
      return this.model.getRequiredOps();
    }
  }

  getSubgraphsSummary() {
    if (this.model._backend !== 'WebML' &&
        this.model &&
        this.model._compilation &&
        this.model._compilation._preparedModel) {
      return this.model._compilation._preparedModel.getSubgraphsSummary();
    } else {
      return [];
    }
  }

  async predict(imageSource) {
    if (!this.initialized) return;
    this.inCtx.drawImage(imageSource, 0, 0,
                         this.inCanvas.width,
                         this.inCanvas.height);
    this.prepareInputTensor(this.inputTensor, this.inCanvas);
    let start = performance.now();
    let result = await this.model.compute([this.inputTensor], [this.outputTensor]);
    let elapsed = performance.now() - start;
    return {time: elapsed.toFixed(2)};
  }

  async loadModelFile(modelUrl) {
    let arrayBuffer = await this.loadUrl(modelUrl, true, true);
    let bytes = new Uint8Array(arrayBuffer);
    return bytes;
  }

  async loadUrl(url, binary, progress) {
    return new Promise((resolve, reject) => {
      if (this.outstandingRequest) {
        this.outstandingRequest.abort();
      }
      let request = new XMLHttpRequest();
      this.outstandingRequest = request;
      request.open('GET', url, true);
      if (binary) {
        request.responseType = 'arraybuffer';
      }
      request.onload = function(ev) {
        this.outstandingRequest = null;
        if (request.readyState === 4) {
          if (request.status === 200) {
              resolve(request.response);
          } else {
              reject(new Error('Failed to load ' + url + ' status: ' + request.status));
          }
        }
      };
      if (progress && typeof this.updateProgress !== 'undefined') {
        request.onprogress = this.updateProgress;
      }
      request.send();
    });
  }

  // uint8 [0, 255] => float [0, 1]
  prepareInputTensor(tensor, canvas) {
    const height = this.inputSize[2];
    const width = this.inputSize[3];
    const [mean, offset] = [255.0, 0.0];
    if (canvas.width !== width || canvas.height !== height) {
      throw new Error(`canvas.width(${canvas.width}) is not ${width} or canvas.height(${canvas.height}) is not ${height}`);
    }
    const ctx = canvas.getContext('2d');
    const pixels = ctx.getImageData(0, 0, width, height).data;
    for(let i = 0; i < height * width; i++){
        // RGB => YUV
        let R = pixels[i * 4];
        let G = pixels[i * 4 + 1];
        let B = pixels[i * 4 + 2];
        let A = pixels[i * 4 + 3];
        let Y = R * 0.256789 + G * 0.504129 + B * 0.097906 + 16;
        let U = R * (-0.148223) + G * (-0.290992) + B * 0.439215;
        let V = R * 0.439215 + G * (-0.367789) + B * (-0.071426);
        tensor[i] = Y / mean -offset;
        this.uData[i] = U;
        this.vData[i] = V;
      }
  }

  drawInput(canvas, imageElement) {
    if (imageElement.width) {
      canvas.width = imageElement.width / imageElement.height * canvas.height;
    } else {
      canvas.width = imageElement.videoWidth / imageElement.videoHeight * canvas.height;
    }
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageElement, 0, 0, canvas.width, canvas.height);
  }

  // float [0, 1] =>  uint8 [0, 255]
  drawOutput(canvas, imageElement) {
    const height = this.outputSize[2];
    const width = this.outputSize[3];
    const [mean, offset] = [255.0, 0.0];
    const srDataArray = new Uint8ClampedArray(width * height * 4);
    for(let i = 0 ; i < this.outputTensor.length; i++) {
      let x = parseInt((i % width) / 2 );
      let y = parseInt((i / width) / 2);
      let index = y * (width / 2) + x;

      let Y = (this.outputTensor[i] + offset ) * mean;
      let U = this.uData[index];
      let V = this.vData[index];
      //YUV => RGB
      srDataArray[i * 4] = (Y - 16) * 1.164383 + V * 1.596027;
      srDataArray[i * 4 + 1] = (Y - 16) * 1.164383 + U * (-0.391762) + V * (-0.812968);
      srDataArray[i * 4 + 2] = (Y - 16) * 1.164383 + U * 2.017232;
      srDataArray[i * 4 + 3] = 255.0;
    }
    const imageData = new ImageData(srDataArray, width, height);

    if (imageElement.width) {
      canvas.width = imageElement.width / imageElement.height * canvas.height;
    } else {
      canvas.width = imageElement.videoWidth / imageElement.videoHeight * canvas.height;
    }
    this.outCtx.putImageData(imageData, 0, 0);
    const ctx = canvas.getContext('2d'); 
    ctx.drawImage(this.outCanvas, 0, 0, canvas.width, canvas.height);
  }

  product(array) {
    return array.reduce((a, b) => a * b);
  }

  deleteAll() {
    if (this.model._backend != 'WebML') {
      this.model._compilation._preparedModel._deleteAll();
    }
  }

  // for debugging
  async iterateLayers(configs, layerList) {
    if (!this.initialized) return;

    let iterators = [];
    let models = [];
    for (let config of configs) {
      let importer = this.modelFile.split('.').pop() === 'tflite' ? TFliteModelImporter : OnnxModelImporter;
      let model = await new importer({
        isQuantized: this.isQuantized,
        rawModel: this.rawModel,
        backend: config.backend,
        prefer: config.prefer || null,
      });
      iterators.push(model.layerIterator([this.inputTensor], layerList));
      models.push(model);
    }

    while (true) {

      let layerOutputs = [];
      for (let it of iterators) {
        layerOutputs.push(await it.next());
      }

      let refOutput = layerOutputs[0];
      if (refOutput.done) {
        break;
      }

      console.debug(`\n\n\nLayer(${refOutput.value.layerId}) ${refOutput.value.outputName}`);

      for (let i = 0; i < configs.length; ++i) {
        console.debug(`\n${configs[i].backend}:`);
        console.debug(`\n${layerOutputs[i].value.tensor}`);

        if (i > 0) {
          let sum = 0;
          for (let j = 0; j < refOutput.value.tensor.length; j++) {
            sum += Math.pow(layerOutputs[i].value.tensor[j] - refOutput.value.tensor[j], 2);
          }
          let variance = sum / refOutput.value.tensor.length;
          console.debug(`var with ${configs[0].backend}: ${variance}`);
        }
      }
    }

    for (let model of models) {
      if (model._backend !== 'WebML') {
        model._compilation._preparedModel._deleteAll();
      }
    }
  }
}

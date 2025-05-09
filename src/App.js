import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [modelId, setModelId] = useState('');
  const [servedModelName, setServedModelName] = useState('');
  const [maxModelLen, setMaxModelLen] = useState('2048');
  const [unlockMaxLength, setUnlockMaxLength] = useState(false);
  const [gpuMemoryUtilization, setGpuMemoryUtilization] = useState('0.9');
  const [tensorParallelSize, setTensorParallelSize] = useState('1');
  const [unlockTensorParallelSize, setUnlockTensorParallelSize] = useState(false);
  const [enablePrefixCaching, setEnablePrefixCaching] = useState(false);
  const [enableChunkedPrefill, setEnableChunkedPrefill] = useState(false);
  const [quantization, setQuantization] = useState('None');
  const [kvCacheType, setKvCacheType] = useState('auto');
  const [copied, setCopied] = useState(false);
  const [animate, setAnimate] = useState(false);
  const [showScrollBtn, setShowScrollBtn] = useState(false);
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
  const [maxBatchSize, setMaxBatchSize] = useState('256');

  const commandSectionRef = useRef(null);

  // Quantization options, keep this in sync with
  // `aphrodite/quantization/__init__.py`
  const quantizationOptions = [
    'None',
    'aqlm',
    'awq',
    'deepspeedfp',
    'tpu_int8',
    'eetq',
    'fp8',
    'quant_llm',
    'fbgemm_fp8',
    'modelopt',
    'gguf',
    'marlin',
    'gptq_marlin_24',
    'gptq_marlin',
    'awq_marlin',
    'gptq',
    'quip',
    'squeezellm',
    'compressed-tensors',
    'compressed_tensors',
    'bitsandbytes',
    'qqq',
    'hqq',
    'experts_int8',
    'fp2',
    'fp3',
    'fp4',
    'fp5',
    'fp6',
    'fp7',
    'neuron_quant',
    'vptq',
    'ipex',
  ];

  const kvCacheTypes = ['auto', 'fp8', 'fp8_e5m2', 'fp8_e4m3'];

  useEffect(() => {
    setAnimate(true);
  }, []);

  useEffect(() => {
    if (!unlockMaxLength && parseInt(maxModelLen) > 32768) {
      setMaxModelLen('32768');
    }
  }, [unlockMaxLength, maxModelLen]);

  useEffect(() => {
    if (!unlockTensorParallelSize && parseInt(tensorParallelSize) > 8) {
      setTensorParallelSize('8');
    }
  }, [unlockTensorParallelSize, tensorParallelSize]);

  // Handle scroll position to show/hide back to top button
  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 300) {
        setShowScrollBtn(true);
      } else {
        setShowScrollBtn(false);
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth',
    });
  };

  const handleCopy = () => {
    const command = generateCommand();
    navigator.clipboard
      .writeText(command)
      .then(() => {
        setCopied(true);
        setTimeout(() => {
          setCopied(false);
        }, 1500);
      })
      .catch(err => {
        console.error('Failed to copy: ', err);
      });
  };

  const generateCommand = () => {
    let command = `aphrodite run ${modelId}`;

    if (servedModelName !== '') {
      command += ` --served-model-name ${servedModelName}`;
    }

    if (maxModelLen !== '2048') {
      command += ` --max-model-len ${maxModelLen}`;
    }

    if (gpuMemoryUtilization !== '0.9') {
      command += ` --gpu-memory-utilization ${gpuMemoryUtilization}`;
    }

    if (tensorParallelSize !== '1') {
      command += ` --tensor-parallel-size ${tensorParallelSize}`;
    }

    if (enablePrefixCaching) {
      command += ` --enable-prefix-caching`;
    }

    if (enableChunkedPrefill) {
      command += ` --enable-chunked-prefill`;
    }

    if (quantization !== 'None') {
      command += ` --quantization ${quantization}`;
    }

    if (kvCacheType !== 'auto') {
      command += ` --kv-cache-type ${kvCacheType}`;
    }

    if (maxBatchSize !== '256') {
      command += ` --max-num-seqs ${maxBatchSize}`;
    }

    return command;
  };

  return (
    <div className={`App ${animate ? 'animate' : ''}`}>
      <header className="header-bar">
        <div className="header-content">
          <div className="logo">
            <i className="fa-solid fa-terminal"></i>
            Aphrodite Engine Tools
          </div>
        </div>
      </header>

      <main className="page-content">
        <h1 className="page-title">Aphrodite Engine CLI Command Generator</h1>
        <p className="subtitle">
          Generate command line instructions for running models with Aphrodite Engine quickly and
          easily
        </p>

        {/* Command Display Section - Moved to the top */}
        <div className="command-section" ref={commandSectionRef}>
          <h2 className="section-title">Generated Command</h2>
          <div className="command-display sticky-command">
            <div className="command-label">Command:</div>
            <code>{generateCommand()}</code>
          </div>
          <button
            className={`copy-button ${copied ? 'copied' : ''}`}
            onClick={handleCopy}
            disabled={!modelId.trim()}
          >
            <i className={copied ? 'fa-solid fa-check' : 'fa-regular fa-clipboard'}></i>
            {copied ? 'Copied!' : 'Copy to Clipboard'}
          </button>
        </div>

        {/* Form Fields Section */}
        <div className="generator-section">
          <div className="form-area">
            <h2 className="section-title">Configure Your Command</h2>

            {/* Model ID field */}
            <div className="input-group">
              <label htmlFor="modelId">
                <i className="fa-solid fa-wand-magic-sparkles"></i> Model ID
              </label>
              <input
                type="text"
                id="modelId"
                value={modelId}
                onChange={e => setModelId(e.target.value)}
                placeholder="Enter model ID (e.g. openai-community/gpt2)"
              />
            </div>

            {/* Served Model Name field */}
            <div className="input-group">
              <label htmlFor="servedModelName">
                <i className="fa-solid fa-server"></i> Model Name on the API (Optional)
              </label>
              <input
                type="text"
                id="servedModelName"
                value={servedModelName}
                onChange={e => setServedModelName(e.target.value)}
              />
              <div className="field-hint">
                The name of the model on the API. Default is the model ID.
              </div>
            </div>

            {/* Max Model Length field */}
            <div className="input-group">
              <div className="input-header">
                <label htmlFor="maxModelLen">
                  <i className="fa-solid fa-arrows-left-right"></i> Max Model Length
                </label>
                <div className="unlock-checkbox">
                  <input
                    type="checkbox"
                    id="unlockMaxLength"
                    checked={unlockMaxLength}
                    onChange={e => setUnlockMaxLength(e.target.checked)}
                  />
                  <label htmlFor="unlockMaxLength" className="checkbox-label">
                    Unlock
                  </label>
                </div>
              </div>
              <div className="slider-container">
                <input
                  type="range"
                  id="maxModelLenSlider"
                  min="256"
                  max={unlockMaxLength ? '2097152' : '32768'}
                  step={unlockMaxLength ? '4096' : '256'}
                  value={maxModelLen}
                  onChange={e => setMaxModelLen(e.target.value)}
                  className="slider"
                />
                <div className="slider-value">{maxModelLen}</div>
              </div>
              <div className="field-hint">
                Maximum sequence length of the model. Defaults to model's max length.{' '}
                {unlockMaxLength && (
                  <span className="warning-text">Large values may increase memory usage.</span>
                )}
              </div>
            </div>

            {/* GPU Memory Utilization */}
            <div className="input-group">
              <label htmlFor="gpuMemoryUtilization">
                <i className="fa-solid fa-memory"></i> GPU Memory Utilization
              </label>
              <div className="slider-container">
                <input
                  type="range"
                  id="gpuMemoryUtilizationSlider"
                  min="0"
                  max="100"
                  step="1"
                  value={Math.round(parseFloat(gpuMemoryUtilization) * 100)}
                  onChange={e =>
                    setGpuMemoryUtilization((parseFloat(e.target.value) / 100).toFixed(2))
                  }
                  className="slider"
                />
                <div className="slider-value">
                  {Math.round(parseFloat(gpuMemoryUtilization) * 100)}%
                </div>
              </div>
              <div className="field-hint">
                The percentage of total GPU memory to use for Aphrodite. Default is 90%. Note that
                Aphrodite Engine will reserve the specified amount of memory, and you will not be
                able to use it for other purposes.
              </div>
            </div>

            {/* Tensor Parallel Size */}
            <div className="input-group">
              <div className="input-header">
                <label htmlFor="tensorParallelSize">
                  <i className="fa-solid fa-computer"></i> Tensor Parallel Size
                </label>
                <div className="unlock-checkbox">
                  <input
                    type="checkbox"
                    id="unlockTensorParallelSize"
                    checked={unlockTensorParallelSize}
                    onChange={e => setUnlockTensorParallelSize(e.target.checked)}
                  />
                  <label htmlFor="unlockTensorParallelSize" className="checkbox-label">
                    Unlock
                  </label>
                </div>
              </div>
              <div className="slider-container">
                <input
                  type="range"
                  id="tensorParallelSizeSlider"
                  min="1"
                  max={unlockTensorParallelSize ? '256' : '8'}
                  step="1"
                  value={tensorParallelSize}
                  onChange={e => setTensorParallelSize(e.target.value)}
                  className="slider"
                />
                <div className="slider-value">{tensorParallelSize}</div>
              </div>
              <div className="field-hint">
                The number of GPUs to use for parallel inference. Default is 1.
                {unlockTensorParallelSize && (
                  <span className="warning-text">
                    If you're doing multi-node inference, please see the{' '}
                    <a href="https://aphrodite.pygmalion.chat/usage/distributed/">here</a>.
                  </span>
                )}
              </div>
            </div>

            {/* Enable Prefix Caching */}
            <div className="input-group">
              <div
                className="toggle-field"
                onClick={() => setEnablePrefixCaching(!enablePrefixCaching)}
              >
                <label className="toggle-label">
                  <i className="fa-solid fa-database"></i> Enable Prefix Caching
                </label>
                <div className="toggle-switch">
                  <input type="checkbox" checked={enablePrefixCaching} readOnly />
                  <span className="toggle-slider"></span>
                </div>
              </div>
              <div className="field-hint">
                Enables caching of previous prompt tokens for better performance with repeated
                queries. Off by default.
              </div>
            </div>

            {/* Enable Chunked Prefill */}
            <div className="input-group">
              <div
                className="toggle-field"
                onClick={() => setEnableChunkedPrefill(!enableChunkedPrefill)}
              >
                <label className="toggle-label">
                  <i className="fa-solid fa-database"></i> Enable Chunked Prefill
                </label>
                <div className="toggle-switch">
                  <input type="checkbox" checked={enableChunkedPrefill} readOnly />
                  <span className="toggle-slider"></span>
                </div>
              </div>
              <div className="field-hint">
                Enables chunked prefill for better memory usage with long sequences. Enabled by
                default for Max Model Length above 16384.
              </div>
            </div>

            {/* Quantization dropdown */}
            <div className="input-group">
              <label htmlFor="quantization">
                <i className="fa-solid fa-microchip"></i> Quantization
              </label>
              <div className="select-container">
                <select
                  id="quantization"
                  value={quantization}
                  onChange={e => setQuantization(e.target.value)}
                  className="select-dropdown fancy-select"
                >
                  {quantizationOptions.map(option => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
                <div className="select-arrow">
                  <i className="fa-solid fa-chevron-down"></i>
                </div>
              </div>
              <div className="field-hint">
                Quantization method to use. Aphrodite usually detects this automatically from the
                model checkpoint. Explicitly setting this may be needed for runtime quantization
                like FP2-FP8 and bitsandbytes.
              </div>
            </div>

            {/* KV Cache Type */}
            <div className="input-group">
              <label htmlFor="kvCacheType">
                <i className="fa-solid fa-database"></i> KV Cache Quantization
              </label>
              <div className="select-container">
                <select
                  id="kvCacheType"
                  value={kvCacheType}
                  onChange={e => setKvCacheType(e.target.value)}
                  className="select-dropdown fancy-select"
                >
                  {kvCacheTypes.map(option => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
                <div className="select-arrow">
                  <i className="fa-solid fa-chevron-down"></i>
                </div>
              </div>
              <div className="field-hint">
                The type of KV cache to use. Defaults to auto (no quantization).
              </div>
            </div>

            {/* Advanced Options Toggle */}
            <div className="input-group advanced-options-toggle">
              <div
                className="toggle-field"
                onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
              >
                <label className="toggle-label">
                  <i className="fa-solid fa-gear"></i> Show Advanced Options
                </label>
                <div className="toggle-switch">
                  <input type="checkbox" checked={showAdvancedOptions} readOnly />
                  <span className="toggle-slider"></span>
                </div>
              </div>
            </div>

            {/* Advanced Options Section */}
            {showAdvancedOptions && (
              <div className="advanced-options-section">
                <div className="advanced-section-header">
                  <h3 className="subsection-title">Advanced Options</h3>
                  <div className="advanced-badge">
                    <i className="fa-solid fa-code"></i> Advanced
                  </div>
                </div>
                <p className="advanced-description">
                  These options provide additional control for expert users. The defaults work well
                  for most cases.
                </p>

                {/* Maximum Batch Size */}
                <div className="input-group">
                  <label htmlFor="maxBatchSize">
                    <i className="fa-solid fa-layer-group"></i> Maximum Batch Size
                  </label>
                  <div className="slider-container">
                    <input
                      type="range"
                      id="maxBatchSizeSlider"
                      min="1"
                      max="1024"
                      step="1"
                      value={maxBatchSize}
                      onChange={e => setMaxBatchSize(e.target.value)}
                      className="slider"
                    />
                    <div className="slider-value">{maxBatchSize}</div>
                  </div>
                  <div className="field-hint">
                    Maximum number of sequences to process in parallel. Default is 256. Higher
                    values may use more GPU memory for CUDA graph captures, or cause OOMs when
                    receiving a large number of requests.
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="footer">
        <div className="footer-content">
          <p>Copyright (c) 2025, PygmalionAI & Ruliad AI</p>
        </div>
      </footer>

      {/* Back to top button */}
      {showScrollBtn && (
        <button className="scroll-to-top-btn" onClick={scrollToTop}>
          <i className="fa-solid fa-arrow-up"></i>
        </button>
      )}
    </div>
  );
}

export default App;

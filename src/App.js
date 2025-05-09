import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [modelId, setModelId] = useState('');
  const [servedModelName, setServedModelName] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [adminKey, setAdminKey] = useState('');
  const [host, setHost] = useState('localhost');
  const [servedModelPort, setServedModelPort] = useState('2242');
  const [maxModelLen, setMaxModelLen] = useState('2048');
  const [unlockMaxLength, setUnlockMaxLength] = useState(false);
  const [gpuMemoryUtilization, setGpuMemoryUtilization] = useState('0.9');
  const [tensorParallelSize, setTensorParallelSize] = useState('1');
  const [unlockTensorParallelSize, setUnlockTensorParallelSize] = useState(false);
  const [enablePrefixCaching, setEnablePrefixCaching] = useState(false);
  const [enableChunkedPrefill, setEnableChunkedPrefill] = useState(false);
  const [quantization, setQuantization] = useState('None');
  const [kvCacheType, setKvCacheType] = useState('auto');
  const [singleUserMode, setSingleUserMode] = useState(false);
  const [numSchedulerSteps, setNumSchedulerSteps] = useState('1');
  const [maxBatchSize, setMaxBatchSize] = useState('256');
  const [pipelineParallelSize, setPipelineParallelSize] = useState('1');
  const [unlockPipelineParallelSize, setUnlockPipelineParallelSize] = useState(false);
  const [launchKoboldAPI, setLaunchKoboldAPI] = useState(false);
  const [cpuOffloadGB, setCpuOffloadGB] = useState('0');
  const [enforceEager, setEnforceEager] = useState(false);
  const [trustRemoteCode, setTrustRemoteCode] = useState(false);
  const [dType, setDType] = useState('auto');
  const [downloadDir, setDownloadDir] = useState('');
  const [loadFormat, setLoadFormat] = useState('auto');
  const [tokenizer, setTokenizer] = useState('auto');
  const [tokenizerMode, setTokenizerMode] = useState('auto');
  const [enableLora, setEnableLora] = useState(false);
  const [maxLoras, setMaxLoras] = useState('1');
  const [maxLoraRank, setMaxLoraRank] = useState('16');
  const [unlockMaxLoraRank, setUnlockMaxLoraRank] = useState(false);
  const [loraExtraVocabSize, setLoraExtraVocabSize] = useState('256');
  const [loraDtype, setLoraDtype] = useState('auto');
  const [longLoraScalingFactors, setLongLoraScalingFactors] = useState('');
  const [fullyShardedLoras, setFullyShardedLoras] = useState(false);
  const [qloraAdapterNameOrPath, setQloraAdapterNameOrPath] = useState('');
  const [loraModules, setLoraModules] = useState([{ key: '', value: '' }]);
  const [enablePromptAdapter, setEnablePromptAdapter] = useState(false);
  const [promptAdapters, setPromptAdapters] = useState([{ key: '', value: '' }]);
  const [maxPromptAdapters, setMaxPromptAdapters] = useState('1');
  const [maxPromptAdapterToken, setMaxPromptAdapterToken] = useState('0');
  const [schedulingPolicy, setSchedulingPolicy] = useState('fcfs');
  const [speculativeModel, setSpeculativeModel] = useState('');
  const [speculativeModelQuantization, setSpeculativeModelQuantization] = useState('None');
  const [numSpeculativeTokens, setNumSpeculativeTokens] = useState('0');
  const [speculativeDisableMQAScorer, setSpeculativeDisableMQAScorer] = useState(false);
  const [speculativeMaxModelLen, setSpeculativeMaxModelLen] = useState('2048');
  const [unlockSpeculativeMaxModelLen, setUnlockSpeculativeMaxModelLen] = useState(false);
  const [speculativePromptLookupMax, setSpeculativePromptLookupMax] = useState('0');
  const [speculativePromptLookupMin, setSpeculativePromptLookupMin] = useState('0');
  const [speculativeDraftTensorParallelSize, setSpeculativeDraftTensorParallelSize] = useState('1');
  const [speculativeDisableByBatchSize, setSpeculativeDisableByBatchSize] = useState('0');
  const [specDecodingAcceptanceMethod, setSpecDecodingAcceptanceMethod] =
    useState('rejection_sampler');
  const [
    typicalAcceptanceSamplerPosteriorThreshold,
    setTypicalAcceptanceSamplerPosteriorThreshold,
  ] = useState('0.09');
  const [typicalAcceptanceSamplerPosteriorAlpha, setTypicalAcceptanceSamplerPosteriorAlpha] =
    useState('0.3');
  const [disableLogprobsDuringSpecDecoding, setDisableLogprobsDuringSpecDecoding] =
    useState('auto');
  const [disableCustomAllReduce, setDisableCustomAllReduce] = useState(false);
  const [allowInlineModelLoading, setAllowInlineModelLoading] = useState(false);

  const [showScrollBtn, setShowScrollBtn] = useState(false);
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
  const [copied, setCopied] = useState(false);
  const [animate, setAnimate] = useState(false);

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

  useEffect(() => {
    if (!unlockPipelineParallelSize && parseInt(pipelineParallelSize) > 8) {
      setPipelineParallelSize('8');
    }
  }, [unlockPipelineParallelSize, pipelineParallelSize]);

  useEffect(() => {
    if (!unlockMaxLoraRank && parseInt(maxLoraRank) > 128) {
      setMaxLoraRank('128');
    }
  }, [unlockMaxLoraRank, maxLoraRank]);

  useEffect(() => {
    if (!unlockSpeculativeMaxModelLen && parseInt(speculativeMaxModelLen) > 32768) {
      setSpeculativeMaxModelLen('32768');
    }
  }, [unlockSpeculativeMaxModelLen, speculativeMaxModelLen]);

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

  const addLoraModule = () => {
    setLoraModules([...loraModules, { key: '', value: '' }]);
  };

  const removeLoraModule = index => {
    const newModules = [...loraModules];
    newModules.splice(index, 1);
    setLoraModules(newModules.length ? newModules : [{ key: '', value: '' }]);
  };

  const updateLoraModule = (index, field, value) => {
    const newModules = [...loraModules];
    newModules[index][field] = value;
    setLoraModules(newModules);
  };

  const getLoraModulesString = () => {
    return loraModules
      .filter(module => module.key.trim() && module.value.trim())
      .map(module => `${module.key.trim()}="${module.value.trim()}"`)
      .join(' ');
  };

  const addPromptAdapter = () => {
    setPromptAdapters([...promptAdapters, { key: '', value: '' }]);
  };

  const removePromptAdapter = index => {
    const newAdapters = [...promptAdapters];
    newAdapters.splice(index, 1);
    setPromptAdapters(newAdapters.length ? newAdapters : [{ key: '', value: '' }]);
  };

  const updatePromptAdapter = (index, field, value) => {
    const newAdapters = [...promptAdapters];
    newAdapters[index][field] = value;
    setPromptAdapters(newAdapters);
  };

  const getPromptAdaptersString = () => {
    return promptAdapters
      .filter(adapter => adapter.key.trim() && adapter.value.trim())
      .map(adapter => `${adapter.key.trim()}="${adapter.value.trim()}"`)
      .join(' ');
  };

  const generateCommand = () => {
    let command = `aphrodite run ${modelId}`;

    if (servedModelName !== '') {
      command += ` --served-model-name ${servedModelName}`;
    }

    if (apiKey !== '') {
      command += ` --api-keys ${apiKey}`;
    }

    if (adminKey !== '') {
      command += ` --admin-key ${adminKey}`;
    }

    if (host !== 'localhost') {
      command += ` --host ${host}`;
    }

    if (servedModelPort !== '2242') {
      command += ` --port ${servedModelPort}`;
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

    if (singleUserMode) {
      command += ` --single-user-mode`;
    }

    if (numSchedulerSteps !== '1') {
      command += ` --num-scheduler-steps ${numSchedulerSteps}`;
    }

    if (maxBatchSize !== '256') {
      command += ` --max-num-seqs ${maxBatchSize}`;
    }

    if (pipelineParallelSize !== '1') {
      command += ` --pipeline-parallel-size ${pipelineParallelSize}`;
    }

    if (launchKoboldAPI) {
      command += ` --launch-kobold-api`;
    }

    if (cpuOffloadGB !== '0') {
      command += ` --cpu-offload-gb ${cpuOffloadGB}`;
    }

    if (enforceEager) {
      command += ` --enforce-eager`;
    }

    if (trustRemoteCode) {
      command += ` --trust-remote-code`;
    }

    if (dType !== 'auto') {
      command += ` --d-type ${dType}`;
    }

    if (loadFormat !== 'auto') {
      command += ` --load-format ${loadFormat}`;
    }

    if (tokenizer !== 'auto' && tokenizer !== '') {
      command += ` --tokenizer ${tokenizer}`;
    }

    if (tokenizerMode !== 'auto') {
      command += ` --tokenizer-mode ${tokenizerMode}`;
    }

    if (enableLora) {
      command += ` --enable-lora`;
    }

    if (maxLoras !== '1') {
      command += ` --max-loras ${maxLoras}`;
    }

    if (maxLoraRank !== '16') {
      command += ` --max-lora-rank ${maxLoraRank}`;
    }

    if (loraExtraVocabSize !== '256') {
      command += ` --lora-extra-vocab-size ${loraExtraVocabSize}`;
    }

    if (loraDtype !== 'auto') {
      command += ` --lora-dtype ${loraDtype}`;
    }

    if (longLoraScalingFactors !== '') {
      command += ` --long-lora-scaling-factors ${longLoraScalingFactors}`;
    }

    if (fullyShardedLoras) {
      command += ` --fully-sharded-loras`;
    }

    if (qloraAdapterNameOrPath !== '') {
      command += ` --qlora-adapter-name-or-path ${qloraAdapterNameOrPath}`;
    }

    const loraModulesString = getLoraModulesString();
    if (loraModulesString) {
      command += ` --lora-modules ${loraModulesString}`;
    }

    if (enablePromptAdapter) {
      command += ` --enable-prompt-adapter`;
    }

    const promptAdaptersString = getPromptAdaptersString();
    if (promptAdaptersString) {
      command += ` --prompt-adapters ${promptAdaptersString}`;
    }

    if (maxPromptAdapters !== '1') {
      command += ` --max-prompt-adapters ${maxPromptAdapters}`;
    }

    if (maxPromptAdapterToken !== '0') {
      command += ` --max-prompt-adapter-token ${maxPromptAdapterToken}`;
    }

    if (schedulingPolicy !== 'fcfs') {
      command += ` --scheduling-policy ${schedulingPolicy}`;
    }

    if (speculativeModel !== '') {
      command += ` --speculative-model ${speculativeModel}`;
    }

    if (speculativeModelQuantization !== 'None') {
      command += ` --speculative-model-quantization ${speculativeModelQuantization}`;
    }

    if (numSpeculativeTokens !== '0') {
      command += ` --num-speculative-tokens ${numSpeculativeTokens}`;
    }

    if (speculativeDisableMQAScorer) {
      command += ` --speculative-disable-mqa-scorer`;
    }

    if (speculativeMaxModelLen !== '2048') {
      command += ` --speculative-max-model-len ${speculativeMaxModelLen}`;
    }

    if (speculativePromptLookupMax !== '0') {
      command += ` --speculative-prompt-lookup-max ${speculativePromptLookupMax}`;
    }

    if (speculativePromptLookupMin !== '0') {
      command += ` --speculative-prompt-lookup-min ${speculativePromptLookupMin}`;
    }

    if (speculativeDraftTensorParallelSize !== '1') {
      command += ` --speculative-draft-tensor-parallel-size ${speculativeDraftTensorParallelSize}`;
    }

    if (speculativeDisableByBatchSize !== '0') {
      command += ` --speculative-disable-by-batch-size ${speculativeDisableByBatchSize}`;
    }

    if (specDecodingAcceptanceMethod !== 'rejection_sampler') {
      command += ` --spec-decoding-acceptance-method ${specDecodingAcceptanceMethod}`;
    }

    if (typicalAcceptanceSamplerPosteriorThreshold !== '0.09') {
      command += ` --typical-acceptance-sampler-posterior-threshold ${typicalAcceptanceSamplerPosteriorThreshold}`;
    }

    if (typicalAcceptanceSamplerPosteriorAlpha !== '0.3') {
      command += ` --typical-acceptance-sampler-posterior-alpha ${typicalAcceptanceSamplerPosteriorAlpha}`;
    }

    if (disableLogprobsDuringSpecDecoding !== 'auto') {
      command += ` --disable-logprobs-during-spec-decoding ${disableLogprobsDuringSpecDecoding}`;
    }

    if (disableCustomAllReduce) {
      command += ` --disable-custom-all-reduce`;
    }

    if (allowInlineModelLoading) {
      command += ` --allow-inline-model-loading`;
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
              <div className="field-hint">
                The ID of the model to run. This is usually the model name on HuggingFace. Can also
                be a local path to a model checkpoint.
              </div>
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

            {/* API Key field */}
            <div className="input-group">
              <label htmlFor="apiKey">
                <i className="fa-solid fa-key"></i> API Key (Optional)
              </label>
              <input
                type="text"
                id="apiKey"
                value={apiKey}
                onChange={e => setApiKey(e.target.value)}
              />
              <div className="field-hint">Custom API key for the API server. Optional.</div>
            </div>

            {/* Admin Key field */}
            <div className="input-group">
              <label htmlFor="adminKey">
                <i className="fa-solid fa-key"></i> Admin Key (Optional)
              </label>
              <input
                type="text"
                id="adminKey"
                value={adminKey}
                onChange={e => setAdminKey(e.target.value)}
              />
              <div className="field-hint">
                Custom admin key for the API server. Optional. If an API key is provided, this
                becomes mandatory to perform some operations, such as inline model loading,
                unload/loading models and adapters.
              </div>
            </div>

            {/* API Server Port field */}
            <div className="input-group">
              <label htmlFor="servedModelPort">
                <i className="fa-solid fa-server"></i> API Server Port
              </label>
              <input
                type="text"
                id="servedModelPort"
                value={servedModelPort}
                onChange={e => setServedModelPort(e.target.value)}
                placeholder="Enter port number (e.g. 2242)"
              />
              <div className="field-hint">
                The port number to serve the model on. Default is 2242.
              </div>
            </div>

            {/* Host for the server */}
            <div className="input-group">
              <label htmlFor="host">
                <i className="fa-solid fa-server"></i> Host
              </label>
              <input
                type="text"
                id="host"
                value={host}
                onChange={e => setHost(e.target.value)}
                placeholder="Enter host (e.g. localhost)"
              />
              <div className="field-hint">
                The host to serve the model on. Default is localhost.
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
                like FP2-FP8 and bitsandbytes. For BitsandBytes, make sure you set the load format
                (Advanced Options) to bitsandbytes.
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

            {/* Single User Mode */}
            <div className="input-group">
              <div className="toggle-field" onClick={() => setSingleUserMode(!singleUserMode)}>
                <label className="toggle-label">
                  <i className="fa-solid fa-user"></i> Single User Mode
                </label>
                <div className="toggle-switch">
                  <input type="checkbox" checked={singleUserMode} readOnly />
                  <span className="toggle-slider"></span>
                </div>
              </div>
              <div className="field-hint">
                Enables single user mode, which allocates enough memory for a single request, and
                disables batching.
              </div>
            </div>

            {/* Number of Scheduler Steps */}
            <div className="input-group">
              <label htmlFor="numSchedulerSteps">
                <i className="fa-solid fa-clock"></i> Number of Scheduler Steps
              </label>
              <div className="slider-container">
                <input
                  type="range"
                  id="numSchedulerStepsSlider"
                  min="1"
                  max="32"
                  step="1"
                  value={numSchedulerSteps}
                  onChange={e => setNumSchedulerSteps(e.target.value)}
                  className="slider"
                />
                <div className="slider-value">{numSchedulerSteps}</div>
              </div>
              <div className="field-hint">
                The number of steps for multi-step scheduling. Default is 1. Higher values will
                enable multi-step scheduling, which may improve throughput.
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

                {/* Pipeline Parallel Size */}
                <div className="input-group">
                  <div className="input-header">
                    <label htmlFor="pipelineParallelSize">
                      <i className="fa-solid fa-computer"></i> Pipeline Parallel Size
                    </label>
                    <div className="unlock-checkbox">
                      <input
                        type="checkbox"
                        id="unlockPipelineParallelSize"
                        checked={unlockPipelineParallelSize}
                        onChange={e => setUnlockPipelineParallelSize(e.target.checked)}
                      />
                      <label htmlFor="unlockPipelineParallelSize" className="checkbox-label">
                        Unlock
                      </label>
                    </div>
                  </div>
                  <div className="slider-container">
                    <input
                      type="range"
                      id="pipelineParallelSizeSlider"
                      min="1"
                      max={unlockPipelineParallelSize ? '256' : '8'}
                      step="1"
                      value={pipelineParallelSize}
                      onChange={e => setPipelineParallelSize(e.target.value)}
                      className="slider"
                    />
                    <div className="slider-value">{pipelineParallelSize}</div>
                  </div>
                  <div className="field-hint">
                    The number of Pipeline Parallel stages. Default is 1. If used together with
                    Tensor Parallelism, the total number of GPUs used will be tensor_parallel_size *
                    pipeline_parallel_size.
                    {unlockPipelineParallelSize && (
                      <span className="warning-text">
                        <br />
                        If you're doing multi-node inference, please see the{' '}
                        <a href="https://aphrodite.pygmalion.chat/usage/distributed/">here</a>.
                      </span>
                    )}
                  </div>
                </div>

                {/* Launch Kobold API*/}
                <div className="input-group">
                  <div
                    className="toggle-field"
                    onClick={() => setLaunchKoboldAPI(!launchKoboldAPI)}
                  >
                    <label className="toggle-label">
                      <i className="fa-solid fa-server"></i> Launch Kobold API
                    </label>
                    <div className="toggle-switch">
                      <input type="checkbox" checked={launchKoboldAPI} readOnly />
                      <span className="toggle-slider"></span>
                    </div>
                  </div>
                  <div className="field-hint">
                    Launch the Kobold API alongside the default OpenAI API.
                  </div>
                </div>

                {/* CPU Offload */}
                <div className="input-group">
                  <label htmlFor="cpuOffloadGB">
                    <i className="fa-solid fa-microchip"></i> CPU Offload
                  </label>
                  <div className="slider-container">
                    <input
                      type="range"
                      id="cpuOffloadGBSlider"
                      min="0.0"
                      max="200.0"
                      step="0.1"
                      value={cpuOffloadGB}
                      onChange={e => setCpuOffloadGB(e.target.value)}
                      className="slider"
                    />
                    <div className="slider-value">{cpuOffloadGB}</div>
                  </div>
                  <div className="field-hint">
                    Amount of virtual memory to add to the GPU using CPU memory. Default is 0.
                    Essentially CPU offloading of the model weights.
                  </div>
                </div>

                {/* Enforce Eager */}
                <div className="input-group">
                  <div className="toggle-field" onClick={() => setEnforceEager(!enforceEager)}>
                    <label className="toggle-label">
                      <i className="fa-solid fa-microchip"></i> Enforce Eager
                    </label>
                    <div className="toggle-switch">
                      <input type="checkbox" checked={enforceEager} readOnly />
                      <span className="toggle-slider"></span>
                    </div>
                  </div>
                  <div className="field-hint">
                    Disable CUDA graph captures. This may slightly reduce memory usage at the cost
                    of performance.
                  </div>
                </div>

                {/* Trust Remote Code */}
                <div className="input-group">
                  <div
                    className="toggle-field"
                    onClick={() => setTrustRemoteCode(!trustRemoteCode)}
                  >
                    <label className="toggle-label">
                      <i className="fa-solid fa-microchip"></i> Trust Remote Code
                    </label>
                    <div className="toggle-switch">
                      <input type="checkbox" checked={trustRemoteCode} readOnly />
                      <span className="toggle-slider"></span>
                    </div>
                  </div>
                  <div className="field-hint">
                    Trust HuggingFace models with custom code. Aphrodite Engine does not use the
                    custom code, but transformers will complain without this.
                  </div>
                </div>

                {/* DType */}
                <div className="input-group">
                  <label htmlFor="dType">
                    <i className="fa-solid fa-microchip"></i> DType
                  </label>
                  <div className="select-container">
                    <select
                      id="dType"
                      value={dType}
                      onChange={e => setDType(e.target.value)}
                      className="select-dropdown fancy-select"
                    >
                      <option value="auto">Auto</option>
                      <option value="bfloat16">BFloat16</option>
                      <option value="float16">Float16</option>
                      <option value="float32">Float32</option>
                    </select>
                    <div className="select-arrow">
                      <i className="fa-solid fa-chevron-down"></i>
                    </div>
                  </div>
                  <div className="field-hint">
                    The data type to use for the model. Default is auto. You usually don't need to
                    change this, unless you have a quantized model that cannot run with bfloat16.
                  </div>
                </div>

                {/* Download Directory */}
                <div className="input-group">
                  <label htmlFor="downloadDir">
                    <i className="fa-solid fa-download"></i> Download Directory
                  </label>
                  <input
                    type="text"
                    id="downloadDir"
                    value={downloadDir}
                    onChange={e => setDownloadDir(e.target.value)}
                    placeholder="Enter download directory"
                  />
                  <div className="field-hint">
                    The directory to download the model to. Default is the HuggingFace cache
                    directory.
                  </div>
                </div>

                {/* Load Format */}
                <div className="input-group">
                  <label htmlFor="loadFormat">
                    <i className="fa-solid fa-file-arrow-down"></i> Load Format
                  </label>
                  <div className="select-container">
                    <select
                      id="loadFormat"
                      value={loadFormat}
                      onChange={e => setLoadFormat(e.target.value)}
                      className="select-dropdown fancy-select"
                    >
                      <option value="auto">Auto</option>
                      <option value="safetensors">SafeTensors</option>
                      <option value="pt">PyTorch</option>
                      <option value="npcache">Numpy Cache</option>
                      <option value="tensorizer">Tensorizer</option>
                      <option value="dummy">Dummy</option>
                      <option value="sharded_state">Sharded State</option>
                      <option value="gguf">GGUF</option>
                      <option value="bitsandbytes">BitsandBytes</option>
                      <option value="mistral">Mistral</option>
                    </select>
                    <div className="select-arrow">
                      <i className="fa-solid fa-chevron-down"></i>
                    </div>
                  </div>
                  <div className="field-hint">
                    The format to load the model in. Default is auto. Use BitsandBytes if you've
                    selected the BitsandBytes quantization method. Dummy can be used to load a model
                    with random weights for testing.
                  </div>
                </div>

                {/* Tokenizer text field*/}
                <div className="input-group">
                  <label htmlFor="tokenizer">
                    <i className="fa-solid fa-microchip"></i> Tokenizer
                  </label>
                  <input
                    type="text"
                    id="tokenizer"
                    value={tokenizer}
                    onChange={e => setTokenizer(e.target.value)}
                    placeholder="Enter custom tokenizer path (optional)"
                  />
                  <div className="field-hint">
                    The tokenizer to use for the model. Leave as "auto" to use the model's default
                    tokenizer. Only set this if you need a different tokenizer than what comes with
                    the model. For GGUF models, this helps with faster loading times if you point it
                    to an unquantized model.
                  </div>
                </div>

                {/* Tokenizer Mode */}
                <div className="input-group">
                  <label htmlFor="tokenizerMode">
                    <i className="fa-solid fa-microchip"></i> Tokenizer Mode
                  </label>
                  <div className="select-container">
                    <select
                      id="tokenizerMode"
                      value={tokenizerMode}
                      onChange={e => setTokenizerMode(e.target.value)}
                      className="select-dropdown fancy-select"
                    >
                      <option value="auto">Auto</option>
                      <option value="slow">Slow</option>
                      <option value="mistral">Mistral</option>
                    </select>
                    <div className="select-arrow">
                      <i className="fa-solid fa-chevron-down"></i>
                    </div>
                  </div>
                  <div className="field-hint">
                    The mode to use for the tokenizer. Default is auto. Use Mistral for Mistral
                    models.
                  </div>
                </div>

                {/* Enable Lora */}
                <div className="input-group">
                  <div className="toggle-field" onClick={() => setEnableLora(!enableLora)}>
                    <label className="toggle-label">
                      <i className="fa-solid fa-microchip"></i> Enable Lora
                    </label>
                    <div className="toggle-switch">
                      <input type="checkbox" checked={enableLora} readOnly />
                      <span className="toggle-slider"></span>
                    </div>
                  </div>
                  <div className="field-hint">Enable handling of LoRA adapters.</div>
                </div>

                {/* LoRA Modules */}
                <div className="input-group lora-modules-group">
                  <label>
                    <i className="fa-solid fa-puzzle-piece"></i> LoRA Modules
                  </label>
                  <div className="field-hint mb-2">
                    Add LoRA modules with their identifiers and paths. Each key is your desired LoRA
                    name, and each value is the HuggingFace ID or local path.
                  </div>

                  {loraModules.map((module, index) => (
                    <div className="lora-module-row" key={index}>
                      <div className="lora-module-inputs">
                        <div className="lora-field-section">
                          <input
                            type="text"
                            placeholder="LoRA name"
                            value={module.key}
                            onChange={e => updateLoraModule(index, 'key', e.target.value)}
                            className={`input-field lora-key ${module.key.trim() ? 'is-valid' : ''}`}
                          />
                        </div>
                        <div className="lora-field-section">
                          <input
                            type="text"
                            placeholder="Path or HF ID"
                            value={module.value}
                            onChange={e => updateLoraModule(index, 'value', e.target.value)}
                            className={`input-field lora-value ${module.value.trim() ? 'is-valid' : ''}`}
                          />
                          <button
                            className="lora-remove-btn"
                            onClick={() => removeLoraModule(index)}
                            type="button"
                            aria-label="Remove LoRA module"
                          >
                            <i className="fa-solid fa-times"></i>
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}

                  <button className="lora-add-btn" onClick={addLoraModule} type="button">
                    <i className="fa-solid fa-plus"></i> Add LoRA Module
                  </button>

                  {getLoraModulesString() && (
                    <div className="lora-preview">
                      <div className="lora-preview-title">Preview:</div>
                      <code className="lora-preview-code">
                        --lora-modules {getLoraModulesString()}
                      </code>
                    </div>
                  )}
                </div>

                {/* Max Loras */}
                <div className="input-group">
                  <label htmlFor="maxLoras">
                    <i className="fa-solid fa-microchip"></i> Max Loras
                  </label>
                  <div className="slider-container">
                    <input
                      type="range"
                      id="maxLorasSlider"
                      min="1"
                      max="1024"
                      step="1"
                      value={maxLoras}
                      onChange={e => setMaxLoras(e.target.value)}
                      className="slider"
                    />
                    <div className="slider-value">{maxLoras}</div>
                  </div>
                  <div className="field-hint">
                    The maximum number of LoRA adapters to load. Default is 1.
                  </div>
                </div>

                {/* Max Lora Rank */}
                <div className="input-group">
                  <div className="input-header">
                    <label htmlFor="maxLoraRank">
                      <i className="fa-solid fa-microchip"></i> Max Lora Rank
                    </label>
                    <div className="unlock-checkbox">
                      <input
                        type="checkbox"
                        id="unlockMaxLoraRank"
                        checked={unlockMaxLoraRank}
                        onChange={e => setUnlockMaxLoraRank(e.target.checked)}
                      />
                      <label htmlFor="unlockMaxLoraRank" className="checkbox-label">
                        Unlock
                      </label>
                    </div>
                  </div>
                  <div className="slider-container">
                    <input
                      type="range"
                      id="maxLoraRankSlider"
                      min="1"
                      max={unlockMaxLoraRank ? '4096' : '128'}
                      step="1"
                      value={maxLoraRank}
                      onChange={e => setMaxLoraRank(e.target.value)}
                      className="slider"
                    />
                    <div className="slider-value">{maxLoraRank}</div>
                  </div>
                  <div className="field-hint">
                    The maximum rank of the LoRA adapters to load. Default is 16.
                    <br />
                    {unlockMaxLoraRank && (
                      <span className="warning-text">
                        Increasing this will increase memory usage and may cause slowdowns.
                      </span>
                    )}
                  </div>
                </div>

                {/* Lora Extra Vocab Size */}
                <div className="input-group">
                  <label htmlFor="loraExtraVocabSize">
                    <i className="fa-solid fa-microchip"></i> Lora Extra Vocab Size
                  </label>
                  <div className="slider-container">
                    <input
                      type="range"
                      id="loraExtraVocabSizeSlider"
                      min="0"
                      max="1024"
                      step="1"
                      value={loraExtraVocabSize}
                      onChange={e => setLoraExtraVocabSize(e.target.value)}
                      className="slider"
                    />
                    <div className="slider-value">{loraExtraVocabSize}</div>
                  </div>
                  <div className="field-hint">
                    The size of the extra vocabulary to load. Default is 256.
                  </div>
                </div>

                {/* Lora dtype */}
                <div className="input-group">
                  <label htmlFor="loraDtype">
                    <i className="fa-solid fa-microchip"></i> Lora DType
                  </label>
                  <div className="select-container">
                    <select
                      id="loraDtype"
                      value={loraDtype}
                      onChange={e => setLoraDtype(e.target.value)}
                      className="select-dropdown fancy-select"
                    >
                      <option value="auto">Auto</option>
                      <option value="bfloat16">BFloat16</option>
                      <option value="float16">Float16</option>
                      <option value="float32">Float32</option>
                    </select>
                    <div className="select-arrow">
                      <i className="fa-solid fa-chevron-down"></i>
                    </div>
                  </div>
                  <div className="field-hint">
                    The data type to use for the LoRA adapters. Default is auto which uses the
                    model's dtype.
                  </div>
                </div>

                {/* Long Lora Scaling Factors */}
                <div className="input-group">
                  <label htmlFor="longLoraScalingFactors">
                    <i className="fa-solid fa-microchip"></i> Long Lora Scaling Factors
                  </label>
                  <div className="input-container">
                    <input
                      type="text"
                      id="longLoraScalingFactors"
                      value={longLoraScalingFactors}
                      onChange={e => setLongLoraScalingFactors(e.target.value)}
                      className="input-field"
                    />
                  </div>
                  <div className="field-hint">
                    Specify multiple scaling factors (which can be different from base model scaling
                    factor - see e.g. Long LoRA) to allow for multiple LoRA adapters trained with
                    those scaling factors to be used at the same time. If not specified, only
                    adapters trained with the base model scaling factors are allowed.
                  </div>
                </div>

                {/* Fully Sharded Loras */}
                <div className="input-group">
                  <div
                    className="toggle-field"
                    onClick={() => setFullyShardedLoras(!fullyShardedLoras)}
                  >
                    <label className="toggle-label">
                      <i className="fa-solid fa-microchip"></i> Fully Sharded Loras
                    </label>
                    <div className="toggle-switch">
                      <input type="checkbox" checked={fullyShardedLoras} readOnly />
                      <span className="toggle-slider"></span>
                    </div>
                  </div>
                  <div className="field-hint">
                    By default, only half of the LoRA computation is sharded with Tensor
                    Parallelism. Enabling this will use the fully sharded layers. At high sequence
                    lengths, max rank or TP size, this is likely faster. Use this with caution, as
                    it may cause slowdowns under most circumstances.
                  </div>
                </div>

                {/* Qlora Adapter Name or Path */}
                <div className="input-group">
                  <label htmlFor="qloraAdapterNameOrPath">
                    <i className="fa-solid fa-microchip"></i> Qlora Adapter Name or Path
                  </label>
                  <input
                    type="text"
                    id="qloraAdapterNameOrPath"
                    value={qloraAdapterNameOrPath}
                    onChange={e => setQloraAdapterNameOrPath(e.target.value)}
                    className="input-field"
                  />
                  <div className="field-hint">The name or path of the Qlora adapter to load.</div>
                </div>

                {/* Enable Prompt Adapter */}
                <div className="input-group">
                  <div
                    className="toggle-field"
                    onClick={() => setEnablePromptAdapter(!enablePromptAdapter)}
                  >
                    <label className="toggle-label">
                      <i className="fa-solid fa-microchip"></i> Enable Prompt Adapter
                    </label>
                    <div className="toggle-switch">
                      <input type="checkbox" checked={enablePromptAdapter} readOnly />
                      <span className="toggle-slider"></span>
                    </div>
                  </div>
                  <div className="field-hint">Enable handling of prompt adapters.</div>
                </div>

                {/* Prompt Adapter Modules */}
                <div className="input-group prompt-adapter-modules-group">
                  <label>
                    <i className="fa-solid fa-puzzle-piece"></i> Prompt Adapter Modules
                  </label>
                  <div className="field-hint mb-2">
                    Add prompt adapter modules with their identifiers and paths. Each key is your
                    desired prompt adapter name, and each value is the HuggingFace ID or local path.
                  </div>

                  {promptAdapters.map((adapter, index) => (
                    <div className="prompt-adapter-module-row" key={index}>
                      <div className="prompt-adapter-module-inputs">
                        <div className="prompt-adapter-field-section">
                          <input
                            type="text"
                            placeholder="Prompt adapter name"
                            value={adapter.key}
                            onChange={e => updatePromptAdapter(index, 'key', e.target.value)}
                            className={`input-field prompt-adapter-key ${adapter.key.trim() ? 'is-valid' : ''}`}
                          />
                        </div>
                        <div className="prompt-adapter-field-section">
                          <input
                            type="text"
                            placeholder="Path or HF ID"
                            value={adapter.value}
                            onChange={e => updatePromptAdapter(index, 'value', e.target.value)}
                            className={`input-field prompt-adapter-value ${adapter.value.trim() ? 'is-valid' : ''}`}
                          />
                          <button
                            className="prompt-adapter-remove-btn"
                            onClick={() => removePromptAdapter(index)}
                            type="button"
                            aria-label="Remove prompt adapter module"
                          >
                            <i className="fa-solid fa-times"></i>
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}

                  <button
                    className="prompt-adapter-add-btn"
                    onClick={addPromptAdapter}
                    type="button"
                  >
                    <i className="fa-solid fa-plus"></i> Add Prompt Adapter Module
                  </button>

                  {getPromptAdaptersString() && (
                    <div className="prompt-adapter-preview">
                      <div className="prompt-adapter-preview-title">Preview:</div>
                      <code className="prompt-adapter-preview-code">
                        --prompt-adapters {getPromptAdaptersString()}
                      </code>
                    </div>
                  )}
                </div>

                {/* Max Prompt Adapters */}
                <div className="input-group">
                  <label>
                    <i className="fa-solid fa-cubes"></i> Max Prompt Adapters
                  </label>
                  <div className="slider-container">
                    <input
                      type="range"
                      value={maxPromptAdapters}
                      onChange={e => setMaxPromptAdapters(e.target.value)}
                      min="1"
                      max="1024"
                      step="1"
                      className="slider"
                    />
                    <div className="slider-value">{maxPromptAdapters}</div>
                  </div>
                  <div className="field-hint">
                    Maximum number of prompt adapters to use (default: 1).
                  </div>
                </div>

                {/* Max Prompt Adapter Token */}
                <div className="input-group">
                  <label>
                    <i className="fa-solid fa-key"></i> Max Prompt Adapter Token
                  </label>
                  <div className="slider-container">
                    <input
                      type="range"
                      value={maxPromptAdapterToken}
                      onChange={e => setMaxPromptAdapterToken(e.target.value)}
                      min="0"
                      max="2048"
                      step="1"
                      className="slider"
                    />
                    <div className="slider-value">{maxPromptAdapterToken}</div>
                  </div>
                  <div className="field-hint">
                    Maximum token length for prompt adapters. 0 means unlimited (default: 0).
                  </div>
                </div>

                {/* Scheduling Policy */}
                <div className="input-group">
                  <label>
                    <i className="fa-solid fa-cubes"></i> Scheduling Policy
                  </label>
                  <div className="select-container">
                    <select
                      value={schedulingPolicy}
                      onChange={e => setSchedulingPolicy(e.target.value)}
                      className="select-dropdown fancy-select"
                    >
                      <option value="fcfs">First Come First Serve</option>
                      <option value="priority">Priority</option>
                    </select>
                    <div className="select-arrow">
                      <i className="fa-solid fa-chevron-down"></i>
                    </div>
                  </div>
                  <div className="field-hint">
                    The scheduling policy to use for the prompt adapters. Default is FCFS. If
                    Priority is enabled, you can specify the priority of a request by adding a
                    `priority` key to the request. 1 is the highest priority, and 0 means no
                    priority.
                  </div>
                </div>

                {/* Speculative Model */}
                <div className="input-group">
                  <label>
                    <i className="fa-solid fa-microchip"></i> Speculative Model
                  </label>
                  <input
                    type="text"
                    value={speculativeModel}
                    onChange={e => setSpeculativeModel(e.target.value)}
                    className="input-field"
                  />
                  <div className="field-hint">
                    The model to use for speculative decoding. Can be an HF ID or a local path. To
                    use an ngram model, put "[ngram]" in the name.
                  </div>
                </div>

                {/* Speculative Model Quantization */}
                <div className="input-group">
                  <label htmlFor="speculativeModelQuantization">
                    <i className="fa-solid fa-microchip"></i> Speculative Model Quantization
                  </label>
                  <div className="select-container">
                    <select
                      id="speculativeModelQuantization"
                      value={speculativeModelQuantization}
                      onChange={e => setSpeculativeModelQuantization(e.target.value)}
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
                    Quantization method to use for the speculative model. This can be different from
                    the main model's quantization.
                  </div>
                </div>

                {/* Number of Speculative Tokens */}
                <div className="input-group">
                  <label>
                    <i className="fa-solid fa-microchip"></i> Number of Speculative Tokens
                  </label>
                  <div className="slider-container">
                    <input
                      type="range"
                      value={numSpeculativeTokens}
                      onChange={e => setNumSpeculativeTokens(e.target.value)}
                      min="0"
                      max="1024"
                      step="1"
                      className="slider"
                    />
                    <div className="slider-value">{numSpeculativeTokens}</div>
                  </div>
                  <div className="field-hint">
                    The number of speculative tokens to use for the speculative model.
                  </div>
                </div>

                {/* Speculative Disable MQA Scorer */}
                <div className="input-group">
                  <div
                    className="toggle-field"
                    onClick={() => setSpeculativeDisableMQAScorer(!speculativeDisableMQAScorer)}
                  >
                    <label className="toggle-label">
                      <i className="fa-solid fa-microchip"></i> Speculative Disable MQA Scorer
                    </label>
                    <div className="toggle-switch">
                      <input type="checkbox" checked={speculativeDisableMQAScorer} readOnly />
                      <span className="toggle-slider"></span>
                    </div>
                  </div>
                  <div className="field-hint">
                    Disable the MQA scorer for speculative decoding and use batch expansion. The MQA
                    scorer currently doesn't work with CUDA graphs so this is recommended.
                  </div>
                </div>

                {/* Speculative Max Model Length */}
                <div className="input-group">
                  <div className="input-header">
                    <label htmlFor="speculativeMaxModelLen">
                      <i className="fa-solid fa-arrows-left-right"></i> Speculative Max Model Length
                    </label>
                    <div className="unlock-checkbox">
                      <input
                        type="checkbox"
                        id="unlockSpeculativeMaxModelLen"
                        checked={unlockSpeculativeMaxModelLen}
                        onChange={e => setUnlockSpeculativeMaxModelLen(e.target.checked)}
                      />
                      <label htmlFor="unlockSpeculativeMaxModelLen" className="checkbox-label">
                        Unlock
                      </label>
                    </div>
                  </div>
                  <div className="slider-container">
                    <input
                      type="range"
                      id="speculativeMaxModelLenSlider"
                      min="256"
                      max={unlockSpeculativeMaxModelLen ? '2097152' : '32768'}
                      step={unlockSpeculativeMaxModelLen ? '4096' : '256'}
                      value={speculativeMaxModelLen}
                      onChange={e => setSpeculativeMaxModelLen(e.target.value)}
                      className="slider"
                    />
                    <div className="slider-value">{speculativeMaxModelLen}</div>
                  </div>
                  <div className="field-hint">
                    Maximum sequence length of the speculative model. Defaults to provided model's
                    max length.
                    {unlockSpeculativeMaxModelLen && (
                      <span className="warning-text"> Large values may increase memory usage.</span>
                    )}
                  </div>
                </div>

                {/* Speculative Prompt Lookup Min */}
                <div className="input-group">
                  <label>
                    <i className="fa-solid fa-search"></i> Speculative Prompt Lookup Min
                  </label>
                  <div className="slider-container">
                    <input
                      type="range"
                      value={speculativePromptLookupMin}
                      onChange={e => setSpeculativePromptLookupMin(e.target.value)}
                      min="0"
                      max="2048"
                      step="1"
                      className="slider"
                    />
                    <div className="slider-value">{speculativePromptLookupMin}</div>
                  </div>
                  <div className="field-hint">
                    Minimum size of window for ngram prompt lookup in speculative decoding. Only
                    applicable to ngram speculative decoding.
                  </div>
                </div>

                {/* Speculative Prompt Lookup Max */}
                <div className="input-group">
                  <label>
                    <i className="fa-solid fa-search"></i> Speculative Prompt Lookup Max
                  </label>
                  <div className="slider-container">
                    <input
                      type="range"
                      value={speculativePromptLookupMax}
                      onChange={e => setSpeculativePromptLookupMax(e.target.value)}
                      min="0"
                      max="2048"
                      step="1"
                      className="slider"
                    />
                    <div className="slider-value">{speculativePromptLookupMax}</div>
                  </div>
                  <div className="field-hint">
                    Maximum size of window for ngram prompt lookup in speculative decoding. Only
                    applicable to ngram speculative decoding.
                  </div>
                </div>

                {/* Speculative Draft Tensor Parallel Size */}
                <div className="input-group">
                  <label>
                    <i className="fa-solid fa-computer"></i> Speculative Draft Tensor Parallel Size
                  </label>
                  <div className="slider-container">
                    <input
                      type="range"
                      value={speculativeDraftTensorParallelSize}
                      onChange={e => setSpeculativeDraftTensorParallelSize(e.target.value)}
                      min="1"
                      max="8"
                      step="1"
                      className="slider"
                    />
                    <div className="slider-value">{speculativeDraftTensorParallelSize}</div>
                  </div>
                  <div className="field-hint">
                    Tensor parallel size for the speculative draft model. Default is 1.
                  </div>
                </div>

                {/* Speculative Disable By Batch Size */}
                <div className="input-group">
                  <label>
                    <i className="fa-solid fa-ban"></i> Disable Speculation By Batch Size
                  </label>
                  <div className="slider-container">
                    <input
                      type="range"
                      value={speculativeDisableByBatchSize}
                      onChange={e => setSpeculativeDisableByBatchSize(e.target.value)}
                      min="0"
                      max="256"
                      step="1"
                      className="slider"
                    />
                    <div className="slider-value">{speculativeDisableByBatchSize}</div>
                  </div>
                  <div className="field-hint">
                    Speculative decoding is disabled for incoming requests if the number of enqueued
                    requests is larger than this value. Default is 0 (disabled).
                  </div>
                </div>

                {/* Speculative Decoding Acceptance Method */}
                <div className="input-group">
                  <label>
                    <i className="fa-solid fa-check-double"></i> Speculative Acceptance Method
                  </label>
                  <div className="select-container">
                    <select
                      value={specDecodingAcceptanceMethod}
                      onChange={e => setSpecDecodingAcceptanceMethod(e.target.value)}
                      className="select-dropdown fancy-select"
                    >
                      <option value="rejection_sampler">Rejection Sampler</option>
                      <option value="typical_acceptance_sampler">Typical Acceptance Sampler</option>
                    </select>
                    <div className="select-arrow">
                      <i className="fa-solid fa-chevron-down"></i>
                    </div>
                  </div>
                  <div className="field-hint">
                    Acceptance method to use during draft token verification in speculative
                    decoding. Two types of acceptance routines are supported: RejectionSampler which
                    does not allow changing the acceptance rate of draft tokens, and
                    TypicalAcceptanceSampler which is configurable, allowing for a higher acceptance
                    rate at the cost of lower quality, and vice versa.
                  </div>
                </div>

                {/* Typical Acceptance Sampler Posterior Threshold */}
                <div className="input-group">
                  <label>
                    <i className="fa-solid fa-sliders"></i> Typical Acceptance Posterior Threshold
                  </label>
                  <div className="slider-container">
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.01"
                      value={typicalAcceptanceSamplerPosteriorThreshold}
                      onChange={e => setTypicalAcceptanceSamplerPosteriorThreshold(e.target.value)}
                      className="slider"
                    />
                    <div className="slider-value">{typicalAcceptanceSamplerPosteriorThreshold}</div>
                  </div>
                  <div className="field-hint">
                    Set the lower bound threshold for the posterior probability of a token to be
                    accepted. This threshold is used by the TypicalAcceptanceSampler to make
                    sampling decisions during speculative decoding. Default is 0.09.
                  </div>
                </div>

                {/* Typical Acceptance Sampler Posterior Alpha */}
                <div className="input-group">
                  <label>
                    <i className="fa-solid fa-sliders"></i> Typical Acceptance Posterior Alpha
                  </label>
                  <div className="slider-container">
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.01"
                      value={typicalAcceptanceSamplerPosteriorAlpha}
                      onChange={e => setTypicalAcceptanceSamplerPosteriorAlpha(e.target.value)}
                      className="slider"
                    />
                    <div className="slider-value">{typicalAcceptanceSamplerPosteriorAlpha}</div>
                  </div>
                  <div className="field-hint">
                    A scaling factor for the entropy-based threshold for token acceptance in the
                    TypicalAcceptanceSampler. Typically defaults to square root of the posterior
                    threshold, i.e. 0.3.
                  </div>
                </div>

                {/* Disable Logprobs During Speculative Decoding */}
                <div className="input-group">
                  <label>
                    <i className="fa-solid fa-calculator"></i> Disable Logprobs During Speculative
                    Decoding
                  </label>
                  <div className="select-container">
                    <select
                      value={disableLogprobsDuringSpecDecoding}
                      onChange={e => setDisableLogprobsDuringSpecDecoding(e.target.value)}
                      className="select-dropdown fancy-select"
                    >
                      <option value="auto">Auto</option>
                      <option value="True">True</option>
                      <option value="False">False</option>
                    </select>
                    <div className="select-arrow">
                      <i className="fa-solid fa-chevron-down"></i>
                    </div>
                  </div>
                  <div className="field-hint">
                    If enabled, token logprobs are not returned during speculative decoding. If not,
                    logprobs are returned according to the settings in SamplingParams. If not
                    specified, it defaults to True. Disabling log probabilities during speculative
                    decoding reduces latency by skipping logprob calculation in proposal sampling,
                    target sampling, and after accepted tokens are determined.
                  </div>
                </div>

                {/* Disable Custom All Reduce toggle*/}
                <div className="input-group">
                  <div
                    className="toggle-field"
                    onClick={() => setDisableCustomAllReduce(!disableCustomAllReduce)}
                  >
                    <label className="toggle-label">
                      <i className="fa-solid fa-ban"></i> Disable Custom All Reduce
                    </label>
                    <div className="toggle-switch">
                      <input type="checkbox" checked={disableCustomAllReduce} readOnly />
                      <span className="toggle-slider"></span>
                    </div>
                  </div>
                  <div className="field-hint">
                    Disable the custom all reduce kernels. Only disable these if your hardware
                    doesn't support them, and you're getting benign warnings in the logger, or if
                    you have issues with the kernels.
                  </div>
                </div>

                {/* Allow Inline Model Loading */}
                <div className="input-group">
                  <div
                    className="toggle-field"
                    onClick={() => setAllowInlineModelLoading(!allowInlineModelLoading)}
                  >
                    <label className="toggle-label">
                      <i className="fa-solid fa-check-double"></i> Allow Inline Model Loading
                    </label>
                    <div className="toggle-switch">
                      <input type="checkbox" checked={allowInlineModelLoading} readOnly />
                      <span className="toggle-slider"></span>
                    </div>
                  </div>
                </div>
                <div className="field-hint">
                  Allow inline model loading. If enabled, specifying a different model in the
                  request from the loaded model will unload the current one and attempt to load the
                  new one. Useful for testing different models without restarting the server. Note
                  that if your server is behind an API key, you will need to also enable admin key
                  and use that for the request.
                </div>
                {/* End */}
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

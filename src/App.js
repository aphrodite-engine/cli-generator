import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [modelId, setModelId] = useState('');
  const [maxModelLen, setMaxModelLen] = useState('8192');
  const [unlockMaxLength, setUnlockMaxLength] = useState(false);
  const [temperature, setTemperature] = useState('0.8');
  const [seed, setSeed] = useState('42');
  const [enablePrefixCaching, setEnablePrefixCaching] = useState(false);
  const [quantization, setQuantization] = useState('None');
  const [copied, setCopied] = useState(false);
  const [animate, setAnimate] = useState(false);

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

  useEffect(() => {
    setAnimate(true);
  }, []);

  useEffect(() => {
    if (!unlockMaxLength && parseInt(maxModelLen) > 32768) {
      setMaxModelLen('32768');
    }
  }, [unlockMaxLength, maxModelLen]);

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

    if (maxModelLen !== '8192') {
      command += ` --max-model-len ${maxModelLen}`;
    }

    if (temperature !== '0.8') {
      command += ` --temperature ${temperature}`;
    }

    if (seed !== '42') {
      command += ` --seed ${seed}`;
    }

    if (enablePrefixCaching) {
      command += ` --enable-prefix-caching`;
    }

    if (quantization !== 'None') {
      command += ` --quantization ${quantization}`;
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
                Maximum sequence length of the model. Default is 8192.{' '}
                {unlockMaxLength && (
                  <span className="warning-text">Large values may increase memory usage.</span>
                )}
              </div>
            </div>

            {/* Temperature field */}
            <div className="input-group">
              <label htmlFor="temperature">
                <i className="fa-solid fa-temperature-half"></i> Temperature
              </label>
              <div className="slider-container">
                <input
                  type="range"
                  id="temperatureSlider"
                  min="0"
                  max="2"
                  step="0.1"
                  value={temperature}
                  onChange={e => setTemperature(e.target.value)}
                  className="slider"
                />
                <div className="slider-value">{temperature}</div>
              </div>
              <div className="field-hint">
                Controls randomness in text generation. Default is 0.8.
              </div>
            </div>

            {/* Seed field */}
            <div className="input-group">
              <label htmlFor="seed">
                <i className="fa-solid fa-dice"></i> Seed
              </label>
              <div className="slider-container">
                <input
                  type="range"
                  id="seedSlider"
                  min="0"
                  max="1000"
                  step="1"
                  value={seed}
                  onChange={e => setSeed(e.target.value)}
                  className="slider"
                />
                <div className="slider-value">{seed}</div>
              </div>
              <div className="field-hint">
                Random seed for reproducible generation. Default is 42.
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
                Enables caching of prompt prefixes for better performance with repeated queries. Off
                by default.
              </div>
            </div>
          </div>

          <div className="result-area">
            <h2 className="section-title">Generated Command</h2>
            <div className="command-display">
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
        </div>
      </main>

      <footer className="footer">
        <div className="footer-content">
          <p>Use this command in your terminal to run the Aphrodite Engine with your model</p>
        </div>
      </footer>
    </div>
  );
}

export default App;

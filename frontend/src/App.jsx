import { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import * as blazeface from "@tensorflow-models/blazeface";

const MODEL_URL = "/web_model/model.json";
const EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"];
const THOUGHTS = {
  angry: ["Who touched my food?!", "I need a reboot.", "This line is too slow."],
  disgust: ["That smell has a backstory.", "Why does this exist?", "My face says it all."],
  fear: ["Did that chair just move?", "Nope, nope, nope.", "Is the door locked?"],
  happy: ["This coffee is amazing.", "Life is good.", "Hehe, this is fun."],
  sad: ["I should have ordered fries too.", "I miss my bed.", "Rainy day energy."],
  surprise: ["Plot twist!", "Wait, what?", "Did not see that coming."],
  neutral: ["Just vibing.", "Running on vibes.", "Loading thoughts..."]
};
const EMOJI_MAP = {
  angry: "😠",
  disgust: "🤢",
  fear: "😱",
  happy: "😄",
  sad: "😢",
  surprise: "😲",
  neutral: "😐"
};

const THOUGHT_INTERVAL_MS = 2000;
const PREDICTION_INTERVAL_MS = 250;

function getRandomThoughtForEmotion(emotion) {
  const options = THOUGHTS[emotion] || THOUGHTS.neutral;
  const index = Math.floor(Math.random() * options.length);
  return options[index];
}

export default function MindReaderApp() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const bubbleRef = useRef(null);
  const ctxRef = useRef(null);
  const cropCanvasRef = useRef(document.createElement("canvas"));
  const cropCtxRef = useRef(null);
  const streamRef = useRef(null);
  const modelsRef = useRef({
    faceModel: null,
    expressionModel: null,
    modelInputSize: 48,
    modelInputChannels: 1,
    loading: null
  });
  const stateRef = useRef({
    running: false,
    rafId: null,
    lastThoughtAt: 0,
    lastPredictionAt: 0,
    lastEmotion: null,
    currentEmotion: null
  });

  const [status, setStatus] = useState("Camera is off.");
  const [statusError, setStatusError] = useState(false);
  const [running, setRunning] = useState(false);
  const [isBusy, setIsBusy] = useState(false);
  const [thoughtText, setThoughtText] = useState("");
  const [bubbleVisible, setBubbleVisible] = useState(false);
  const [emotion, setEmotion] = useState(null);

  useEffect(() => {
    if (canvasRef.current) {
      ctxRef.current = canvasRef.current.getContext("2d");
    }
    cropCtxRef.current = cropCanvasRef.current.getContext("2d", { willReadFrequently: true });

    return () => {
      stateRef.current.running = false;
      if (stateRef.current.rafId) {
        cancelAnimationFrame(stateRef.current.rafId);
      }
      stopCamera();
    };
  }, []);

  function setStatusMessage(message, isError) {
    setStatus(message);
    setStatusError(Boolean(isError));
  }

  function setEmotionSafe(next) {
    setEmotion((prev) => (prev === next ? prev : next));
  }

  function resizeCanvasToVideo() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (video && canvas && video.videoWidth && video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
  }

  async function loadModels() {
    const models = modelsRef.current;
    if (models.faceModel && models.expressionModel) {
      return models;
    }
    if (models.loading) {
      return models.loading;
    }

    // Load and cache the TF.js models once per session.
    models.loading = (async () => {
      await tf.ready();
      models.faceModel = await blazeface.load();
      try {
        models.expressionModel = await tf.loadLayersModel(MODEL_URL);
      } catch (error) {
        const message = error && error.message ? error.message : String(error);
        if (message.indexOf("404") !== -1) {
          throw new Error(
            `Model not found at ${MODEL_URL}. Place model.json and shard files in public/web_model/ or update MODEL_URL.`
          );
        }
        throw error;
      }

      const inputs = models.expressionModel && models.expressionModel.inputs;
      if (inputs && inputs[0] && inputs[0].shape && inputs[0].shape.length === 4) {
        models.modelInputSize = inputs[0].shape[1] || models.modelInputSize;
        models.modelInputChannels = inputs[0].shape[3] || models.modelInputChannels;
      }

      cropCanvasRef.current.width = models.modelInputSize;
      cropCanvasRef.current.height = models.modelInputSize;
      return models;
    })();

    return models.loading;
  }

  async function initCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("Camera access is not supported in this browser.");
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false
    });

    streamRef.current = stream;
    const video = videoRef.current;
    if (!video) {
      throw new Error("Video element not ready.");
    }

    video.srcObject = stream;
    await video.play();

    if (video.readyState < 2) {
      await new Promise((resolve) => {
        video.onloadedmetadata = () => resolve();
      });
    }

    resizeCanvasToVideo();
  }

  function stopCamera() {
    const stream = streamRef.current;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }

  function hideBubble() {
    setBubbleVisible(false);
  }

  function showBubble() {
    setBubbleVisible(true);
  }

  function positionBubble(box) {
    if (!stateRef.current.currentEmotion) {
      return;
    }

    const bubble = bubbleRef.current;
    const canvas = canvasRef.current;
    if (!bubble || !canvas) {
      return;
    }

    const padding = 12;
    const bubbleWidth = bubble.offsetWidth || 0;
    const bubbleHeight = bubble.offsetHeight || 0;
    let x = box.x + box.width + padding;
    let y = box.y - bubbleHeight - padding;

    if (x + bubbleWidth + padding > canvas.width) {
      x = box.x - bubbleWidth - padding;
    }
    if (x < padding) {
      x = padding;
    }
    if (y < padding) {
      y = box.y + padding;
    }

    bubble.style.left = `${x}px`;
    bubble.style.top = `${y}px`;
  }

  function drawBox(box) {
    const ctx = ctxRef.current;
    if (!ctx) {
      return;
    }
    ctx.strokeStyle = "#38e1b6";
    ctx.lineWidth = 2;
    ctx.strokeRect(box.x, box.y, box.width, box.height);
  }

  function clampBox(box) {
    const canvas = canvasRef.current;
    if (!canvas) {
      return { x: 0, y: 0, width: 0, height: 0 };
    }
    const x = Math.max(0, Math.floor(box.x));
    const y = Math.max(0, Math.floor(box.y));
    const width = Math.min(canvas.width - x, Math.floor(box.width));
    const height = Math.min(canvas.height - y, Math.floor(box.height));
    return { x, y, width, height };
  }

  function getFaceBox(prediction) {
    const x1 = prediction.topLeft[0];
    const y1 = prediction.topLeft[1];
    const x2 = prediction.bottomRight[0];
    const y2 = prediction.bottomRight[1];
    return clampBox({
      x: x1,
      y: y1,
      width: x2 - x1,
      height: y2 - y1
    });
  }

  async function classifyFace(box) {
    if (box.width <= 0 || box.height <= 0) {
      return null;
    }

    const video = videoRef.current;
    const models = modelsRef.current;
    if (!video || !models.expressionModel) {
      return null;
    }

    const cropCtx = cropCtxRef.current;
    if (!cropCtx) {
      return null;
    }

    // Crop the face region and run the expression model on the resized patch.
    cropCtx.clearRect(0, 0, cropCanvasRef.current.width, cropCanvasRef.current.height);
    cropCtx.drawImage(
      video,
      box.x,
      box.y,
      box.width,
      box.height,
      0,
      0,
      models.modelInputSize,
      models.modelInputSize
    );

    const prediction = tf.tidy(() => {
      let input = tf.browser.fromPixels(cropCanvasRef.current).toFloat();
      if (models.modelInputChannels === 1) {
        input = input.mean(2).expandDims(2);
      }
      const normalized = input.div(255);
      const batched = normalized.expandDims(0);
      return models.expressionModel.predict(batched);
    });

    const scores = await prediction.data();
    prediction.dispose();

    let maxIndex = 0;
    for (let i = 1; i < scores.length; i += 1) {
      if (scores[i] > scores[maxIndex]) {
        maxIndex = i;
      }
    }

    return EMOTION_LABELS[maxIndex] || "neutral";
  }

  function updateThought(emotionLabel) {
    // Throttle updates so the bubble doesn't flicker every frame.
    const now = performance.now();
    const tracker = stateRef.current;
    const shouldUpdate =
      emotionLabel !== tracker.lastEmotion || now - tracker.lastThoughtAt > THOUGHT_INTERVAL_MS;

    if (shouldUpdate) {
      setThoughtText(getRandomThoughtForEmotion(emotionLabel));
      tracker.lastThoughtAt = now;
      tracker.lastEmotion = emotionLabel;
    }

    setEmotionSafe(emotionLabel);
    showBubble();
  }

  async function loop() {
    // Main loop: detect a face, classify emotion, update visuals.
    if (!stateRef.current.running) {
      return;
    }

    const video = videoRef.current;
    if (!video || video.readyState < 2) {
      stateRef.current.rafId = requestAnimationFrame(loop);
      return;
    }

    const models = modelsRef.current;
    if (!models.faceModel) {
      stateRef.current.rafId = requestAnimationFrame(loop);
      return;
    }

    const predictions = await models.faceModel.estimateFaces(video, false);
    const ctx = ctxRef.current;
    const canvas = canvasRef.current;
    if (ctx && canvas) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    if (!predictions.length) {
      hideBubble();
      stateRef.current.currentEmotion = null;
      stateRef.current.lastEmotion = null;
      setEmotionSafe(null);
      stateRef.current.rafId = requestAnimationFrame(loop);
      return;
    }

    const faceBox = getFaceBox(predictions[0]);
    drawBox(faceBox);

    const now = performance.now();
    if (now - stateRef.current.lastPredictionAt > PREDICTION_INTERVAL_MS) {
      stateRef.current.lastPredictionAt = now;
      const emotionLabel = await classifyFace(faceBox);
      if (emotionLabel && stateRef.current.running) {
        stateRef.current.currentEmotion = emotionLabel;
        updateThought(emotionLabel);
      }
    }

    positionBubble(faceBox);
    stateRef.current.rafId = requestAnimationFrame(loop);
  }

  async function start() {
    if (stateRef.current.running) {
      return;
    }

    setIsBusy(true);
    setStatusMessage("Loading models...", false);

    try {
      await loadModels();
      setStatusMessage("Starting camera...", false);
      await initCamera();
      stateRef.current.running = true;
      stateRef.current.lastPredictionAt = 0;
      setRunning(true);
      setStatusMessage("Looking for faces...", false);
      loop();
    } catch (error) {
      const message = error && error.message ? error.message : String(error);
      setStatusMessage(`Could not start: ${message}`, true);
      stop({ preserveStatus: true });
    } finally {
      setIsBusy(false);
    }
  }

  function stop(options) {
    const preserveStatus = options && options.preserveStatus;
    stateRef.current.running = false;
    setRunning(false);

    if (stateRef.current.rafId) {
      cancelAnimationFrame(stateRef.current.rafId);
      stateRef.current.rafId = null;
    }

    const ctx = ctxRef.current;
    const canvas = canvasRef.current;
    if (ctx && canvas) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    hideBubble();
    stateRef.current.currentEmotion = null;
    stateRef.current.lastEmotion = null;
    setEmotionSafe(null);
    stopCamera();

    if (!preserveStatus) {
      setStatusMessage("Camera is off.", false);
    }
  }

  return (
    <main className="app">
      <header>
        <h1 className="title">Mind Reader</h1>
        <p className="tagline">A playful face-powered thought bubble.</p>
        <div className="controls">
          <button
            id="toggleCamera"
            type="button"
            onClick={running ? () => stop() : start}
            disabled={isBusy}
          >
            {running ? "Stop Camera" : "Start Camera"}
          </button>
          <span
            id="status"
            className={statusError ? "error" : ""}
            role="status"
            aria-live="polite"
          >
            {status}
          </span>
        </div>
      </header>

      <section id="stage">
        <video
          id="video"
          ref={videoRef}
          autoPlay
          playsInline
          muted
          width="640"
          height="480"
        ></video>
        <canvas id="overlay" ref={canvasRef} width="640" height="480"></canvas>
        <div
          id="thoughtBubble"
          ref={bubbleRef}
          className={bubbleVisible ? "visible" : ""}
          aria-live="polite"
        >
          <span className="bubble-emoji" aria-hidden="true">
            {EMOJI_MAP[emotion] || "🙂"}
          </span>
          <span className="bubble-text">{thoughtText}</span>
        </div>
      </section>

      <div className="emotion-readout" role="status" aria-live="polite">
        <span className="emotion-label">Detected emotion</span>
        <span className="emotion-value">
          <span className="emotion-emoji" aria-hidden="true">
            {emotion ? EMOJI_MAP[emotion] || "🙂" : "👀"}
          </span>
          <span className="emotion-text">{emotion || "No face detected"}</span>
        </span>
      </div>

      <p className="note">
        This app is for fun only - it does NOT actually read minds, it only guesses facial expressions.
      </p>
    </main>
  );
}

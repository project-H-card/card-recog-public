// app_vector.js - DINOv2 Vector Search (ES Module)

import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/dist/transformers.min.js';

const DB_BIN_URL   = '/assets/vector-db.bin';
const DB_INDEX_URL = '/assets/vector-db-index.json';
const DB_JSON_URL  = '/assets/vector-db.json';
const METADATA_URL = '/assets/card-metadata.json';
const MODEL_ID     = 'Xenova/dinov2-small';

// TTA（推論時拡張）: 複数角度で推論して最高スコアを採用
const TTA_ANGLES = [-15, 0, 15];

// 類似度がこの値未満のときは「自信なし」としてTOP-3を提示
const CONFIDENCE_THRESHOLD = 0.65;

// ガイド枠のサイズ比率（CSS側の .card-guide-frame と同期すること）
// カメラ表示 320×320px に対して枠が 180×252px → 56.25% × 78.75%
const GUIDE_W_RATIO = 180 / 320;  // 56.25%
const GUIDE_H_RATIO = 252 / 320;  // 78.75%

let extractor = null;
let vectorDB = { cardIds: [], vectors: [] };
let cardMetadata = {};

// DOM Elements
const statusEl        = document.getElementById('model-status');
const spinnerEl       = document.getElementById('loading-spinner');
const mainContentEl   = document.getElementById('main-content');
const video           = document.getElementById('webcam');
const toggleCameraBtn = document.getElementById('toggle-camera-btn');
const imageUpload     = document.getElementById('image-upload');
const previewArea     = document.getElementById('preview-area');
const previewImage    = document.getElementById('preview-image');
const predictBtn      = document.getElementById('predict-btn');
const resultSection   = document.getElementById('prediction-result');
const resultConfident = document.getElementById('result-confident');
const resultUncertain = document.getElementById('result-uncertain');
const resultCardName  = document.getElementById('result-card-name');
const resultCardId    = document.getElementById('result-card-id');
const confidenceText  = document.getElementById('confidence-text');
const candidatesList  = document.getElementById('candidates-list');
const captureCanvas   = document.getElementById('capture-canvas');
const captureOkMsg    = document.getElementById('capture-ok-msg');
const resultLoading   = document.getElementById('result-loading');

function setStatus(text) {
    statusEl.textContent = text;
    return new Promise(r => requestAnimationFrame(() => setTimeout(r, 0)));
}

let isCameraActive = false;
let currentImageSource = null;

function cosineSimilarity(vecA, vecB) {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dot += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    if (normA === 0 || normB === 0) return 0;
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/** バイナリDB をロードして { cardIds, vectors } に変換する */
async function loadVectorDB() {
    try {
        const [binRes, idxRes] = await Promise.all([
            fetch(DB_BIN_URL),
            fetch(DB_INDEX_URL),
        ]);
        if (binRes.ok && idxRes.ok) {
            const [arrayBuf, idx] = await Promise.all([
                binRes.arrayBuffer(),
                idxRes.json(),
            ]);
            const view = new DataView(arrayBuf);
            const N = view.getUint32(0, true);
            const D = view.getUint32(4, true);
            const vectors = [];
            for (let i = 0; i < N; i++) {
                const offset = 8 + i * D * 4;
                vectors.push(new Float32Array(arrayBuf, offset, D));
            }
            console.log(`Binary DB loaded: ${N} vectors, dim=${D}`);
            return { cardIds: idx.cardIds, vectors };
        }
    } catch (_) {}

    // フォールバック: JSON 形式
    const res = await fetch(DB_JSON_URL);
    const json = await res.json();
    const cardIds = Object.keys(json);
    const vectors = cardIds.map(id => new Float32Array(json[id][0]));
    console.log(`JSON DB loaded: ${cardIds.length} cards`);
    return { cardIds, vectors };
}

async function init() {
    try {
        await setStatus('特徴量DB をロード中... (1/2)');
        vectorDB = await loadVectorDB();

        try {
            const metaResponse = await fetch(METADATA_URL);
            if (metaResponse.ok) cardMetadata = await metaResponse.json();
        } catch (_) {}

        await setStatus('AIモデル をダウンロード中... (2/2)　※初回は少し時間がかかります');
        extractor = await pipeline('image-feature-extraction', MODEL_ID);

        spinnerEl.classList.add('hide');
        statusEl.textContent = '準備完了';
        statusEl.classList.add('ready');
        mainContentEl.classList.remove('hide');
        predictBtn.disabled = false;

    } catch (error) {
        console.error('Error during initialization:', error);
        spinnerEl.classList.add('hide');
        statusEl.textContent = 'ロードに失敗しました。ローカルサーバで起動しているか確認してください。';
        statusEl.style.color = 'red';
    }
}

async function toggleCamera() {
    if (isCameraActive) {
        const stream = video.srcObject;
        if (stream) stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        isCameraActive = false;
        toggleCameraBtn.textContent = 'カメラON';
        currentImageSource = null;
    } else {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment', width: 320, height: 320 }
            });
            video.srcObject = stream;
            isCameraActive = true;
            toggleCameraBtn.textContent = 'カメラOFF';
            previewArea.classList.add('hide');
            currentImageSource = video;
        } catch (err) {
            console.error('Error accessing webcam:', err);
            alert('カメラのアクセスに失敗しました。');
        }
    }
}

imageUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (isCameraActive) toggleCamera();
    const reader = new FileReader();
    reader.onload = (event) => {
        previewImage.src = event.target.result;
        previewArea.classList.remove('hide');
        currentImageSource = previewImage;
    };
    reader.readAsDataURL(file);
});

toggleCameraBtn.addEventListener('click', toggleCamera);

/**
 * 画像ソースから中央クロップした Canvas を返す。
 * CENTER_CROP_RATIO の割合だけ中心を切り出すことで背景の影響を低減する。
 */
function captureSourceCanvas() {
    const src = currentImageSource;
    const sw = (src === video) ? video.videoWidth  : (src.naturalWidth  || src.width);
    const sh = (src === video) ? video.videoHeight : (src.naturalHeight || src.height);

    const cw = Math.floor(sw * GUIDE_W_RATIO);
    const ch = Math.floor(sh * GUIDE_H_RATIO);
    const cx = Math.floor((sw - cw) / 2);
    const cy = Math.floor((sh - ch) / 2);

    const c = document.createElement('canvas');
    c.width  = cw;
    c.height = ch;
    c.getContext('2d').drawImage(src, cx, cy, cw, ch, 0, 0, cw, ch);
    return c;
}

/** Canvas を中心回転させた新しい Canvas を返す */
function rotateCanvas(src, angleDeg) {
    const rad = angleDeg * Math.PI / 180;
    const dst = document.createElement('canvas');
    dst.width  = src.width;
    dst.height = src.height;
    const ctx = dst.getContext('2d');
    ctx.translate(src.width / 2, src.height / 2);
    ctx.rotate(rad);
    ctx.drawImage(src, -src.width / 2, -src.height / 2);
    return dst;
}

/**
 * TTA（Test Time Augmentation）推論
 * 複数角度で推論し、TOP-3 カードを返す。
 */
async function predictWithTTA(srcCanvas) {
    const { cardIds, vectors } = vectorDB;

    // cardIds は重複あり（1カード × N_SAMPLES ベクトル）
    const scores = {};

    for (const angleDeg of TTA_ANGLES) {
        const rotated = rotateCanvas(srcCanvas, angleDeg);
        const dataUrl = rotated.toDataURL('image/jpeg', 0.9);

        const result = await extractor(dataUrl);
        // dims: [1, 257, 384] → CLS token = data[0..dims[2]-1]
        const vec = result.data.slice(0, result.dims[2]);

        for (let i = 0; i < cardIds.length; i++) {
            const sim = cosineSimilarity(vec, vectors[i]);
            const id = cardIds[i];
            if (scores[id] === undefined || sim > scores[id]) scores[id] = sim;
        }
    }

    const top3 = Object.entries(scores)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 3)
        .map(([cardId, similarity]) => ({ cardId, similarity }));

    return top3;
}

function getDisplayName(cardId) {
    const meta = cardMetadata[cardId];
    return meta ? meta.name : cardId;
}

/** キャプチャ済みフレームをオーバーレイ表示してフリーズ感を出す */
function freezeCameraFrame(srcCanvas) {
    captureCanvas.width  = srcCanvas.width;
    captureCanvas.height = srcCanvas.height;
    captureCanvas.getContext('2d').drawImage(srcCanvas, 0, 0);
    captureCanvas.classList.remove('hide');
    captureOkMsg.classList.remove('hide');
}

function unfreezeCameraFrame() {
    captureCanvas.classList.add('hide');
    captureOkMsg.classList.add('hide');
}

predictBtn.addEventListener('click', async () => {
    if (!extractor || !currentImageSource) {
        alert('画像ソースを用意してください。');
        return;
    }

    predictBtn.disabled = true;
    predictBtn.textContent = `解析中 (${TTA_ANGLES.length}方向)...`;

    // キャプチャしてカメラをフリーズ（以降カードを外してOK）
    const srcCanvas = captureSourceCanvas();
    freezeCameraFrame(srcCanvas);

    // 結果エリアをローディング状態に
    resultSection.classList.remove('hide');
    resultLoading.classList.remove('hide');
    resultConfident.classList.add('hide');
    resultUncertain.classList.add('hide');

    try {
        const top3 = await predictWithTTA(srcCanvas);
        const best = top3[0];
        const confident = best.similarity >= CONFIDENCE_THRESHOLD;

        resultLoading.classList.add('hide');

        if (confident) {
            // 高信頼: 1件表示
            resultConfident.classList.remove('hide');
            resultUncertain.classList.add('hide');

            resultCardName.textContent = getDisplayName(best.cardId);
            resultCardId.textContent = best.cardId;
            confidenceText.textContent = best.similarity.toFixed(4);

        } else {
            // 低信頼: TOP-3 候補を提示
            resultConfident.classList.add('hide');
            resultUncertain.classList.remove('hide');

            candidatesList.innerHTML = '';
            top3.forEach((r, i) => {
                const btn = document.createElement('button');
                btn.className = 'candidate-btn';
                btn.innerHTML = `
                    <span class="candidate-rank">${i + 1}</span>
                    <span class="candidate-name">${getDisplayName(r.cardId)}</span>
                    <span class="candidate-sim">${r.similarity.toFixed(3)}</span>`;
                btn.addEventListener('click', () => {
                    resultConfident.classList.remove('hide');
                    resultUncertain.classList.add('hide');
                    resultCardName.textContent = getDisplayName(r.cardId);
                    resultCardId.textContent = r.cardId;
                    confidenceText.textContent = r.similarity.toFixed(4) + ' (手動選択)';
                });
                candidatesList.appendChild(btn);
            });
        }

    } catch (e) {
        console.error('Prediction error:', e);
        alert('推論中にエラーが発生しました。');
    } finally {
        unfreezeCameraFrame();
        predictBtn.disabled = false;
        predictBtn.textContent = '認識する';
    }
});

window.addEventListener('load', init);

"use client";

import { useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";

export default function Home() {
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [result, setResult] = useState<string>("");
  const [confidence, setConfidence] = useState<number | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    const loadModel = async () => {
      console.log("Cargando modelo...");
      const loadedModel = await tf.loadLayersModel("/carpeta_salida/model.json");
      setModel(loadedModel);
      console.log("Modelo cargado correctamente");
    };
    loadModel();
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const handleResetFeedback = () => {
    setResult("");
    setConfidence(null);
    setErrorMessage(null);
  };

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    const newPreview = URL.createObjectURL(file);
    setPreviewUrl(newPreview);
    handleResetFeedback();
  };

  const loadImage = (src: string) =>
    new Promise<HTMLImageElement>((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = (err) => reject(err);
      image.src = src;
    });

  const runPrediction = async () => {
    if (!model) {
      setErrorMessage("El modelo aún no está listo. Intenta nuevamente en unos segundos.");
      return;
    }
    if (!previewUrl) {
      setErrorMessage("Primero selecciona una imagen de perro o gato.");
      return;
    }

    try {
      setIsPredicting(true);
      setErrorMessage(null);

      const imageElement = await loadImage(previewUrl);

      const score = tf.tidy(() => {
        const tensor = tf.browser
          .fromPixels(imageElement, 1) // fuerza 1 canal para coincidir con el modelo
          .resizeNearestNeighbor([100, 100])
          .toFloat()
          .div(255)
          .expandDims();

        const prediction = model.predict(tensor) as tf.Tensor;
        return prediction.dataSync()[0];
      });

      setConfidence(score);
      setResult(score > 0.5 ? "Perro" : "Gato");
    } catch (error) {
      console.error("Error al predecir", error);
      setErrorMessage("No se pudo procesar la imagen. Prueba con otra distinta.");
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <main className="relative flex min-h-screen items-center justify-center bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 px-6 py-16 text-white">
      <div className="absolute inset-0 -z-10 bg-[radial-gradient(circle_at_top,_rgba(56,189,248,0.15),_transparent_60%)]" />
      <div className="max-w-3xl w-full space-y-8 rounded-3xl bg-white/5 p-10 shadow-[0_40px_80px_-40px_rgba(15,23,42,0.8)] backdrop-blur">
        <div className="text-center space-y-3">
          <h1 className="text-4xl font-bold tracking-tight">aPredicción de Perros guau guau vs Gatos miau miau</h1>
          <p className="text-slate-300">
            Subirr foto y presionar en el boton predecir para saber si es guau guau o miau miau
          </p>
        </div>

        <div className="flex items-center justify-center gap-2 text-sm text-slate-400">
          {model ? "Modelo cargado y listo para usar" : "Cargando modelo..."}
        </div>

        <div className="grid gap-8 md:grid-cols-[1.2fr_1fr] md:items-center">
          <div className="space-y-6">
            <label
              htmlFor="image-upload"
              className="flex cursor-pointer flex-col items-center justify-center rounded-2xl border border-dashed border-slate-600 bg-white/5 p-8 text-center transition hover:border-sky-400 hover:shadow-[0_20px_50px_-30px_rgba(56,189,248,0.7)]"
            >
              <input
                id="image-upload"
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden"
              />
              <div className="space-y-3">
                <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full bg-sky-500/20 text-2xl">
                  📷
                </div>
                <p className="text-lg font-semibold text-white">Selecciona una foto</p>
              </div>
            </label>

            <button
              type="button"
              onClick={runPrediction}
              disabled={!model || !previewUrl || isPredicting}
              className="inline-flex w-full items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-sky-500 to-cyan-400 px-6 py-3 text-lg font-semibold text-slate-950 shadow-[0_20px_50px_-20px_rgba(34,211,238,0.7)] transition hover:scale-[1.01] focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isPredicting ? (
                <>
                  <span className="h-5 w-5 animate-spin rounded-full border-2 border-slate-900 border-t-transparent" />
                  Analizando imagen
                </>
              ) : (
                <>
                  Obtener prediccion
                </>
              )}
            </button>

            {errorMessage && (
              <p className="rounded-xl border border-red-400/40 bg-red-500/10 p-3 text-sm text-red-200">
                {errorMessage}
              </p>
            )}
          </div>

          <div className="space-y-4">
            <div className="aspect-square w-full overflow-hidden rounded-2xl border border-slate-700 bg-slate-900/60 shadow-inner">
              {previewUrl ? (
                <img
                  src={previewUrl}
                  alt="Vista previa"
                  className="h-full w-full object-cover transition duration-500 ease-out hover:scale-105"
                />
              ) : (
                <div className="flex h-full flex-col items-center justify-center gap-2 text-slate-500">
                  <p className="text-sm">La imagen aparecera aquí</p>
                </div>
              )}
            </div>

            {result && (
              <div className="rounded-2xl border border-emerald-400/40 bg-emerald-500/10 p-4 text-emerald-200 transition-all">
                <p className="text-xl font-semibold">{result}</p>
                {confidence !== null && (
                  <p className="text-sm text-emerald-200/80">
                    Confianza: {(confidence * 100).toFixed(1)}%
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}

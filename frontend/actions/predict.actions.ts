"use server";

import axios from "axios";

export async function predict(file: File | null) {
  if (!file) return;

  const arrayBuffer = await file.arrayBuffer();
  const blob = new Blob([arrayBuffer], { type: "audio/vnd.wave" });

  const formData = new FormData();
  formData.append("audio", blob, file.name);

  try {
    const response = await axios.post(
      "http://localhost:3333/predict",
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      },
    );

    console.log("Prediction response:", response.data);
    return response.data;
  } catch (error) {
    console.error("Prediction failed:", error);
    throw error;
  }
}

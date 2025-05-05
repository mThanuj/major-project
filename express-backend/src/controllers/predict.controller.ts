import axios from "axios";
import { Request, Response } from "express";
import { Feature } from "../types/feature";

export const predict = async (req: Request, res: Response) => {
  const file = req.file;

  if (!file) {
    res.status(400).json({ error: "No audio file uploaded" });
    return;
  }

  const response = await axios.post(
    "http://localhost:5000/extract-features",
    file.buffer,
    {
      headers: {
        "Content-Type": "audio/wav",
      },
    },
  );

  if (response.status !== 200) {
    res.status(500).json({ error: "Failed to extract features" });
    return;
  }

  const features = response.data.features;

  let keywords: string[] = [];

  features.forEach((feature: Feature) => {
    keywords.push(feature.predicted_label);
  });

  res.status(200).json({
    keywords,
  });

  res.status(200).json({ message: "Audio file uploaded successfully" });
};

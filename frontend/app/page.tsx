"use client";

import React, { useState, useRef } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { predict } from "@/actions/predict.actions";

const page = () => {
  const [file, setFile] = useState<File | null>(null);
  const [keywords, setKeywords] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const [recording, setRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const handlePredict = async (audioFile: File) => {
    setLoading(true);
    setKeywords([]);
    try {
      const data = await predict(audioFile);
      if (data?.keywords) {
        setKeywords(data.keywords);
      }
    } catch (err) {
      console.error("Prediction failed", err);
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
    }
  };

  const handleFileSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!file) return;
    await handlePredict(file);
  };

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
    mediaRecorderRef.current = mediaRecorder;
    audioChunksRef.current = [];

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunksRef.current.push(event.data);
      }
    };

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunksRef.current, { type: "audio/wav" });
      const wavFile = new File([audioBlob], "recording.wav", {
        type: "audio/wav",
      });
      setRecording(false);
      clearInterval(timerRef.current!);
      setRecordingTime(0);
      await handlePredict(wavFile);
    };

    mediaRecorder.start();
    setRecording(true);
    setRecordingTime(0);
    timerRef.current = setInterval(() => {
      setRecordingTime((prev) => prev + 1);
    }, 1000);
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }
  };

  return (
    <motion.div
      className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-br from-purple-600 to-blue-500 p-4 space-y-6"
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
    >
      <div className="flex flex-col md:flex-row gap-6">
        <Card className="w-full max-w-md shadow-2xl rounded-2xl">
          <CardHeader>
            <CardTitle className="text-xl">Upload Audio</CardTitle>
            <CardDescription>Upload a .wav file</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleFileSubmit} className="space-y-4">
              <div>
                <Label htmlFor="wav-upload">.wav File</Label>
                <Input
                  id="wav-upload"
                  type="file"
                  accept=".wav"
                  className="cursor-pointer"
                  onChange={handleFileChange}
                />
              </div>
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Button
                  type="submit"
                  disabled={!file || loading}
                  className="w-full"
                >
                  {loading
                    ? "Processing..."
                    : file
                      ? "Submit File"
                      : "Choose File"}
                </Button>
              </motion.div>
            </form>
          </CardContent>
        </Card>
        <Card className="w-full max-w-md shadow-2xl rounded-2xl">
          <CardHeader>
            <CardTitle className="text-xl">Record Audio</CardTitle>
            <CardDescription>Record and submit your voice</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-col sm:flex-row gap-2">
              <Button
                onClick={startRecording}
                disabled={loading}
                className="w-full sm:w-1/2"
                variant="default"
              >
                🎙️ Start
              </Button>
              <Button
                onClick={stopRecording}
                disabled={loading}
                className="w-full sm:w-1/2"
                variant="destructive"
              >
                ⏹️ Stop & Submit
              </Button>
            </div>
            {recording && (
              <div className="flex items-center gap-2 text-red-600 font-medium">
                <motion.div
                  className="w-3 h-3 rounded-full bg-red-500"
                  animate={{ scale: [1, 1.5, 1] }}
                  transition={{ repeat: Infinity, duration: 1 }}
                />
                <span>
                  Recording... {Math.floor(recordingTime / 60)}:
                  {(recordingTime % 60).toString().padStart(2, "0")}
                </span>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
      {keywords.length > 0 && (
        <motion.div
          className="w-full max-w-md"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Card className="bg-white bg-opacity-90 shadow-lg rounded-2xl p-4">
            <CardHeader>
              <CardTitle className="text-xl">Detected Keywords</CardTitle>
              <CardDescription className="text-slate-50">
                From MFCC + Autoencoder + CNN
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-wrap gap-2">
              {keywords.map((kw, index) => (
                <Badge
                  key={index}
                  variant="secondary"
                  className="text-sm px-3 py-1"
                >
                  {kw}
                </Badge>
              ))}
            </CardContent>
          </Card>
        </motion.div>
      )}
    </motion.div>
  );
};

export default page;

import express from "express";
import { config } from "./utils/config";

import predictRouter from "./routes/predict.route";
import multer from "multer";

const app = express();

const upload = multer({ storage: multer.memoryStorage() });

app.use(express.json());

app.use("/predict", upload.single("audio"), predictRouter);

app.listen(config.PORT, () => {
  console.log("Server listening on port 3000");
});

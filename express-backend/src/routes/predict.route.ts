import express from "express";
import { predict } from "../controllers/predict.controller";

const router = express.Router();

router.post("/", predict);

export default router;

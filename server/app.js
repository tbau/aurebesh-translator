// * TensorFlow Stuff
import * as tf from "@tensorflow/tfjs-node";
import mobilenet from '@tensorflow-models/mobilenet';
import coco_ssd from "@tensorflow-models/coco-ssd";
import fs from 'fs';
import path from 'path';
import sharp from 'sharp';
import * as url from 'url';
const __dirname = url.fileURLToPath(new URL('.',
    import.meta.url));

// * Server Stuff
import express from "express";
import busboy from "busboy";
import { config } from "dotenv";
config();

global.__basedir = __dirname;

// * Init Model
// let model = undefined;
let mobilenetModel = undefined;
// (async() => {

//     mobilenetModel = await mobilenet.load();
//     model = await coco_ssd.load({
//         base: "mobilenet_v1",
//     });
// })();

// * Init Express
const app = express();
const PORT = process.env.PORT || 5000;
app.use(express.json());

app.post("/predict", (req, res) => {
    if (!model) {
        res.status(500).send("Model is not loaded yet!");
        return;
    }

    // * Create a Busboy instance
    const bb = busboy({ headers: req.headers });
    bb.on("file", (fieldname, file, filename, encoding, mimetype) => {
        const buffer = [];
        file.on("data", (data) => {
            buffer.push(data);
        });
        file.on("end", async() => {
            // * Run Object Detection
            const image = tf.node.decodeImage(Buffer.concat(buffer));
            const predictions = await model.detect(image, 3, 0.25);
            res.json(predictions);
        });
    });
    req.pipe(bb);
});

let generateTrainingData = async() => {
    let letters = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
        'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Empty1',
        'X', 'Y', 'Z', 'CH', 'AE', 'EO', 'KH', 'NG', 'OO', 'SH', 'TH', 'Empty2'
    ];

    let numbers = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Empty3', 'Empty4'
    ];

    let symbols = [
        ',', '.', '?', '!', ':', ';', '-', '"', '\'', '(', ')', '/'
    ]

    let lettersFile = await sharp(path.join(__basedir, "../models/aurebesh-model/inputs/aurebesh-letters-1.gif"));
    // let numbersFile = await sharp(path.join(__basedir, "../models/aurebesh-model/inputs/aurebesh-numbers.gif"));
    // let symbolsFile = await sharp(path.join(__basedir, "../models/aurebesh-model/inputs/aurebesh-symbols.gif"));

    let letterWidth = 48;
    let letterHeight = 84;
    let imageWidth = 576;
    let imageHeight = 252;
    for (var i = 0; i * letterWidth < imageWidth; i++) {
        for (var j = 0; j * letterHeight < imageHeight; j++) {
            lettersFile.extract({ width: 48, height: 36, left: i * 48, top: j * 84 }).toFile(path.join(__basedir, "../models/aurebesh-model/training-data/", `${letters[i+j*12]}.png`));
        }
    }
}

let generateValidationData = async() => {
    let letters = [
        'A', 'B', 'C', 'D', 'E', 'F',
        'G', 'H', 'I', 'J', 'K', 'L',
        'M', 'N', 'O', 'P', 'Q', 'R',
        'S', 'T', 'U', 'V', 'W', 'X',
        'Y', 'Z', 'CH', 'AE', 'EO', 'KH',
        'Empty1', 'NG', 'OO', 'SH', 'TH', 'Empty2'
    ];

    let lettersFile = await sharp(path.join(__basedir, "../models/aurebesh-model/inputs/aurebesh-letters-2.jpg"));
    let letterWidth = 114;
    let letterHeight = 116;
    let imageWidth = 684;
    let imageHeight = 696;
    for (var i = 0; i * letterWidth < imageWidth; i++) {
        for (var j = 0; j * letterHeight < imageHeight; j++) {
            lettersFile.extract({ width: letterWidth, height: letterHeight - 50, left: i * letterWidth, top: j * letterHeight }).toFile(path.join(__basedir, "../models/aurebesh-model/validation-data/", `${letters[i+j*6]}.png`));
        }
    }
}

let trainModel = async() => {

    let letters = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
        'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
        'X', 'Y', 'Z', 'CH', 'AE', 'EO', 'KH', 'NG', 'OO', 'SH', 'TH'
    ];

    let trainingTensorFeaturesArray = [];
    let validationTensorFeaturesArray = [];

    for (var l in letters) {
        const imageBuffer = await fs.readFileSync(path.join(__basedir, "../models/aurebesh-model/training-data/", `${letters[l]}.png`));
        let tensorFeature = tf.node.decodeImage(imageBuffer);
        trainingTensorFeaturesArray.push(tensorFeature);
    }


    for (var l in letters) {
        const imageBuffer = await fs.readFileSync(path.join(__basedir, "../models/aurebesh-model/validation-data/", `${letters[l]}.png`));
        let tensorFeature = tf.image.resizeBilinear(tf.node.decodeImage(imageBuffer), [36, 48]);
        validationTensorFeaturesArray.push(tensorFeature);
    }

    let trainingTensorFeatures = tf.stack([...trainingTensorFeaturesArray]);
    let validationTensorFeatures = tf.stack([...validationTensorFeaturesArray]);


    let labelArray = [];
    for (var i = 0; i < 34; i++) {
        labelArray.push(i);
    }

    let tensorLabels = tf.oneHot(tf.tensor1d(labelArray, 'int32'), 34);

    const model = tf.sequential();


    let width = 48;
    let height = 36;

    model.add(tf.layers.conv2d({
        inputShape: [height, width, 3], // numberOfChannels = 3 for colorful images and one otherwise
        filters: 32,
        kernelSize: 3,
        activation: 'relu',
    }));

    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 34, activation: 'sigmoid' }));

    model.compile({
        optimizer: 'sgd',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    await model.fit(trainingTensorFeatures, tensorLabels, {
        epochs: 10000,
        learningRate: 0.1,
        validationData: [validationTensorFeatures, tensorLabels]
    });


    const imageBuffer = await fs.readFileSync(path.join(__basedir, "../models/aurebesh-model/validation-data", `C.png`));
    let tensorFeature = tf.image.resizeBilinear(tf.node.decodeImage(imageBuffer), [36, 48]);

    let result = await model.predict(tf.stack([tensorFeature]));
    console.log(result.dataSync())

}
trainModel();

// app.listen(PORT, () => {
//     console.log(`Server started on port ${PORT}`);
//     // generateTrainingData();
//     // generateValidationData();
// });
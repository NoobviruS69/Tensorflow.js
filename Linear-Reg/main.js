let xdata = [];
let ordinates = [];

let weights, bias;

const lr = 0.5;
const optimizer = tf.train.sgd(lr);

function setup() {
  createCanvas(700,700)
  background(60)

  weights = tf.variable(tf.scalar(random(1)))
  bias = tf.variable(tf.scalar(random(1)))
}

function mousePressed() {
  let x = map(mouseX, 0, width, 0, 1)
  let y = map(mouseY, 0, height, 1, 0)
  xdata.push(x);
  ordinates.push(y);
}

function draw() {

  tf.tidy(() => {
    if (xdata.length > 0) {
      const truey = tf.tensor1d(ordinates)
      optimizer.minimize(() => loss(predict(xdata), truey))
    }
  });

  background(60)
  stroke(255)
  strokeWeight(8)

  for (let i = 0; i < xdata.length; i++) {
    let px = map(xdata[i], 0, 1, 0, width)
    let py = map(ordinates[i], 0, 1, height, 0)
    point(px, py);
  }


  const startX = [0, 1];

  const ys = tf.tidy(() => predict(startX))
  let lineY = ys.dataSync();
  ys.dispose();

  let x1 = map(startX[0], 0, 1, 0, width)
  let x2 = map(startX[1], 0, 1, 0, width)

  let y1 = map(lineY[0], 0, 1, height, 0)
  let y2 = map(lineY[1], 0, 1, height, 0)

  strokeWeight(2);
  line(x1, y1, x2, y2);

}


function loss(pred, labels) {
  return pred.sub(labels).square().mean()
}

function predict(x) {
  const xs = tf.tensor1d(x)
  const predy = xs.mul(weights).add(bias)
  return predy
}
use peroxide::fuga::*;

// =============================================================================
// Main
// =============================================================================
pub const EPOCH: usize = 5;

fn main() {
    // Data Generation
    let train_df = data_gen(100);
    let val_df = data_gen(50);

    // Preprocessing
    let (train_data, train_label) = data_to_vec(&train_df);
    let (val_data, val_label) = data_to_vec(&val_df);

    // Declare Perceptron
    let mut p = Perceptron::new(0.1);

    // Training & Validation
    let mut train_losses = vec![0f64; EPOCH];
    let mut val_losses = vec![0f64; EPOCH];
    let mut val_accs = vec![0f64; EPOCH];

    for epoch in 0 .. EPOCH {
        // Train
        let mut train_loss = 0f64;
        for (xs, t) in train_data.iter().zip(train_label.iter()) {
            train_loss += p.update(xs, *t);
        }

        // Validation
        let mut val_loss = 0f64;
        let mut val_acc = 0f64;
        for (xs, t) in val_data.iter().zip(val_label.iter()) {
            val_loss += p.loss(xs, *t);
            val_acc += p.accuracy(xs, *t) as f64;
        }

        train_loss /= train_data.len() as f64;
        val_loss /= val_data.len() as f64;
        val_acc /= val_data.len() as f64;

        train_losses[epoch] = train_loss;
        val_losses[epoch] = val_loss;
        val_accs[epoch] = val_acc;

        println!("=== Epoch {} ===", epoch);
        println!("train loss: {}", train_loss);
        println!("val loss: {}", val_loss);
        println!("val acc: {}", val_acc);
        println!("");
    }
    
    // // Prediction
    // for (xs, t) in train_data.iter().zip(lab.iter()) {
    //     println!("True: {}, Predict: {}", t, p.forward(xs));
    // }

    // Extract Weights
    let W = p.get_W();
    println!("Weight: {:?}", W);

    // Find the boundary
    let decision_boundary = |x: f64| { -(W[0] + W[1] * x) / W[2] };

    // Prepare for plotting
    let domain = seq(-1f64, 3f64, 0.01);
    let image = domain.fmap(decision_boundary);

    // Plot
    let mut plt = Plot2D::new();
    plt.set_domain(domain)
        .insert_image(image)
        .insert_pair((train_df["x1"].to_vec(), train_df["y1"].to_vec()))
        .insert_pair((train_df["x2"].to_vec(), train_df["y2"].to_vec()))
        .set_title("Perceptron")
        .set_legend(vec!["Boundary", "Group1", "Group2"])
        .set_marker(vec![Line, Point, Point])
        .set_path("plot.png");

    plt.savefig().expect("Error saving plot");
}

// =============================================================================
// Perceptron Struct
// =============================================================================
#[derive(Debug)]
struct Perceptron {
    W: Vec<f64>,
    lr: f64,
}

impl Perceptron {
    pub fn new(lr: f64) -> Self {
        let W = rand(1, 3).data;
        Self {
            W,
            lr,
        }
    }

    pub fn forward(&self, data: &Vec<f64>) -> f64 {
        self.W.dot(data).signum()
    }

    pub fn update(&mut self, data: &Vec<f64>, label: f64) -> f64 {
        let y_hat = self.forward(data);
        let grad = data.mul_s(self.lr * (label - y_hat));
        self.W = self.W.add_v(&grad);

        let loss = (label - y_hat).powi(2);
        loss
    }

    pub fn loss(&self, data: &Vec<f64>, label: f64) -> f64 {
        let y_hat = self.forward(data);
        let loss = (label - y_hat).powi(2);
        loss
    }

    pub fn accuracy(&self, data: &Vec<f64>, label: f64) -> usize {
        let y_hat = self.forward(data);
        let acc = if y_hat == label { 1 } else { 0 };
        acc
    }

    pub fn get_W(&self) -> &Vec<f64> {
        &self.W
    }
}

// =============================================================================
// Data Generation
// =============================================================================
fn data_gen(n: usize) -> DataFrame {
    let mut df = DataFrame::new(vec![]);

    let nx1 = Normal(2.0, 0.5);
    let ny1 = Normal(0.0, 0.5);
    let nx2 = Normal(0.0, 0.25);
    let ny2 = Normal(2.0, 0.25);

    let t1 = vec![1f64; n];
    let t2 = vec![-1f64; n];

    df.push("x1", Series::new(nx1.sample(n)));
    df.push("y1", Series::new(ny1.sample(n)));
    df.push("t1", Series::new(t1));
    df.push("x2", Series::new(nx2.sample(n)));
    df.push("y2", Series::new(ny2.sample(n)));
    df.push("t2", Series::new(t2));

    df
}

fn data_to_vec(df: &DataFrame) -> (Vec<Vec<f64>>, Vec<f64>) {
    let group1 = df["x1"].to_vec()
        .iter()
        .zip(df["y1"].to_vec().iter())
        .map(|(&x, &y)| vec![1f64, x, y])
        .collect::<Vec<Vec<f64>>>();
    let lab1: Vec<f64> = df["t1"].to_vec();

    let group2 = df["x2"].to_vec()
        .iter()
        .zip(df["y2"].to_vec().iter())
        .map(|(&x, &y)| vec![1f64, x, y])
        .collect::<Vec<Vec<f64>>>();
    let lab2: Vec<f64> = df["t2"].to_vec();

    let mut data = group1;
    data.extend(group2.into_iter());

    let mut lab = lab1;
    lab.extend(lab2.into_iter());

    (data, lab)
}
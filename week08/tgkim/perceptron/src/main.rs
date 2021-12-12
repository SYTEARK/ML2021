use peroxide::fuga::*;

// =============================================================================
// Main
// =============================================================================
fn main() {
    // Data Generation
    let df = data_gen();
    df.print();

    // Preprocessing
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

    let mut group = group1;
    group.extend(group2.into_iter());

    let mut lab = lab1;
    lab.extend(lab2.into_iter());

    // Declare Perceptron
    let mut p = Perceptron::new(0.1);

    // Training
    let mut loss = 0f64;
    for (xs, t) in group.iter().zip(lab.iter()) {
        loss = p.update(xs, *t);
        println!("Loss: {}", loss);
    }

    // Prediction
    for (xs, t) in group.iter().zip(lab.iter()) {
        println!("True: {}, Predict: {}", t, p.forward(xs));
    }

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
        .insert_pair((df["x1"].to_vec(), df["y1"].to_vec()))
        .insert_pair((df["x2"].to_vec(), df["y2"].to_vec()))
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

    pub fn get_W(&self) -> &Vec<f64> {
        &self.W
    }
}

// =============================================================================
// Data Generation
// =============================================================================
fn data_gen() -> DataFrame {
    let mut df = DataFrame::new(vec![]);

    let nx1 = Normal(2.0, 0.5);
    let ny1 = Normal(0.0, 0.5);
    let nx2 = Normal(0.0, 0.25);
    let ny2 = Normal(2.0, 0.25);

    let t1 = vec![1f64; 100];
    let t2 = vec![-1f64; 100];

    df.push("x1", Series::new(nx1.sample(100)));
    df.push("y1", Series::new(ny1.sample(100)));
    df.push("t1", Series::new(t1));
    df.push("x2", Series::new(nx2.sample(100)));
    df.push("y2", Series::new(ny2.sample(100)));
    df.push("t2", Series::new(t2));

    df
}

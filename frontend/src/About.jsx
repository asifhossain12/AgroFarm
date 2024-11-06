import React from "react";
import { LuExternalLink } from "react-icons/lu";

const About = () => {
  return (
    <div style={styles.container}>
      <h1 style={styles.title}>About AgroFarm</h1>
      <p style={styles.paragraph}>
        Welcome to AgroFarm, where we are dedicated to revolutionizing the agricultural industry through innovative solutions and sustainable practices. Our mission is to empower farmers, enhance food production, and promote environmental stewardship.
      </p>

      <h2 style={styles.subTitle}>Repository Information</h2>
      <p style={styles.paragraph}>
      Click <a href="https://github.com/asifhossain12/AgroFarm" style={styles.link} target="_blank"><LuExternalLink /></a> to view the complete GitHub repository.
      </p>

      <h2 style={styles.subTitle}>Our Mission</h2>
      <p style={styles.paragraph}>
      In today's world, where technology is rapidly advancing, artificial intelligence (AI) and machine learning (ML) play crucial roles across various sectors. One such initiative focuses on supporting farmers, particularly potato farmers, by providing a platform to detect diseases affecting potato leaves, such as early blight and late blight. Through our website, farmers can easily upload images of their potato leaves, and our deep learning model will analyze the photos to identify the specific disease category with high accuracy. This innovative solution not only empowers farmers with timely information but also contributes to improving crop health and yield, ultimately fostering sustainable agricultural practices.
      </p>
      <h2 style={styles.subTitle}>Concise explanation of Early Blight and Late Blight leaf disease</h2>
      <p style={styles.paragraph}>
      Overview of the two types of potato diseases: Early Blight and Late Blight.
      </p>
      <ul style={styles.list}>
      <li style={styles.listItem}>Early Blight: Caused by the fungus "Alternaria solani", it manifests as dark, concentric leaf spots, yellowing of surrounding tissue, and premature leaf drop, leading to reduced plant vigor.</li>
      <li style={styles.listItem}>Management of Early Blight: Control strategies include crop rotation, planting resistant varieties, and applying fungicides, along with maintaining good cultural practices like proper irrigation and nutrient management.</li>
      <li style={styles.listItem}>Late Blight: Caused by the pathogen "Phytophthora infestans", it presents as water-soaked leaf lesions that rapidly expand, along with potential black lesions on stems and tubers, resulting in quick crop devastation.</li>
      <li style={styles.listItem}>Management of Late Blight: Effective management involves using resistant potato varieties, proper field sanitation, timely fungicide applications, and monitoring weather conditions to implement integrated pest management (IPM) practices.</li>
      </ul>
      <p style={styles.paragraph}>
        For more detailed information, you can check out this article <a href="https://ipm.cahnr.uconn.edu/early-blight-and-late-blight-of-potato/" style={styles.link} target="_blank"><LuExternalLink /></a>
      </p>
      <h2 style={styles.subTitle}>Procedure for Potato Disease Classification Model</h2>
      <p style={styles.paragraph}>
        Overview of the procedure to create your potato disease classification model:
      </p>
      <ul style={styles.list}>
        <li style={styles.listItem}>Dataset Collection: The project utilizes the Plant Village dataset, which contains images of potato leaves exhibiting various diseases, including early blight, late blight, and healthy leaves.</li>
        <li style={styles.listItem}>Train-Test Split: The dataset is divided into training and testing subsets to evaluate the model's performance on unseen data, ensuring generalization capability.</li>
        <li style={styles.listItem}>Data Augmentation: Various techniques such as rotation, flipping, zooming, and shifting are applied to enhance the diversity of the training dataset and prevent overfitting.</li>
        <li style={styles.listItem}>Rescaling: Image pixel values are normalized to a range of 0 to 1 to improve convergence speed during training and enable effective learning.</li>
        <li style={styles.listItem}>CNN Architecture: A Convolutional Neural Network is created with 32 convolutional layers followed by a max pooling layer to learn intricate patterns and features from the images.</li>
        <li style={styles.listItem}>Activation Function: The ReLU (Rectified Linear Unit) activation function is used to introduce non-linearity, allowing the model to learn complex functions.</li>
        <li style={styles.listItem}>Model Training: The CNN model is trained for 21 epochs on the training dataset, adjusting parameters to minimize loss and improve accuracy.</li>
        <li style={styles.listItem}>Model Evaluation: The model achieves approximately 99% accuracy on the training dataset, reflecting its excellent performance in classifying potato leaf diseases.</li>
      </ul>
      <p style={styles.paragraph}>
        Join us in our journey towards a sustainable agricultural future. Together, we can make a difference in the world of farming!
      </p>
      <p style={styles.paragraph}>
        For more detailed information, click <a href="https://github.com/asifhossain12?tab=repositories" style={styles.link} target="_blank"><LuExternalLink /></a>
      </p>
    </div>
  );
};

const styles = {
  container: {
    maxWidth: "800px",
    margin: "0 auto",
    padding: "2rem",
    lineHeight: "1.6",
  },
  title: {
    fontSize: "2.5rem",
    marginBottom: "1rem",
    color: "#4CAF50",
  },
  subTitle: {
    fontSize: "1.8rem",
    margin: "1.5rem 0 0.5rem",
    color: "#4CAF50",
  },
  paragraph: {
    fontSize: "1.1rem",
    marginBottom: "1rem",
  },
  list: {
    marginLeft: "1.5rem",
    marginBottom: "1rem",
  },
  listItem: {
    marginBottom: "0.5rem",
  },
};

export default About;

import React from "react";
import { FiGithub } from "react-icons/fi";
import { IoLogoInstagram } from "react-icons/io5";
import { SlSocialLinkedin } from "react-icons/sl";

const Footer = () => {
  return (
    <footer style={styles.footer}>
      <p style={styles.copyText}>© {new Date().getFullYear()} AgroFarm. All rights reserved.</p>
      <p style={styles.copyText}> Developed by Shaikh Asif Hossain </p>
      <div style={styles.socialLinks}>
        <a href="https://www.linkedin.com/in/asifhossain01/" style={styles.icon} target="_blank" rel="noopener noreferrer">
        <SlSocialLinkedin />
        </a>
        <a href="https://www.instagram.com/_.asif_12/" style={styles.icon} target="_blank" rel="noopener noreferrer">
        <IoLogoInstagram />
        </a>
        <a href="https://github.com/asifhossain12" style={styles.icon} target="_blank" rel="noopener noreferrer">
        <FiGithub />
        </a>
      </div>
    </footer>
  );
};

const styles = {
  footer: {
    backgroundColor: "#4CAF50",
    color: "white",
    padding: "1rem 2rem",
    textAlign: "center",
    marginTop: "auto",
  },
  copyText: {
    margin: "0.5rem 0",
    fontSize: "0.9rem",
  },
  socialLinks: {
    display: "flex",
    justifyContent: "center",
    gap: "1rem",
  },
  icon: {
    color: "white",
    textDecoration: "none",
    fontSize: "1rem",
  },
};

export default Footer;

import React from "react";
import { Link } from "react-router-dom";
import logo from "./aglogo.PNG"


const Navbar = () => {
  return (
    <nav style={styles.navbar}>
      <img src={logo} alt="AgroFarm Logo" style={styles.logo} />
      <div style={styles.navLinks}>
        <Link to="/" style={styles.link}>Home</Link>
        <Link to="/about" style={styles.link}>About</Link>
      </div>
    </nav>
  );
};

const styles = {
  navbar: {
    display: "flex",
    justifyContent: "flex-start", 
    alignItems: "center",
    padding: "1rem 2rem",
    backgroundColor: "#4CAF50",
    color: "white",
    width: "100%",
    position: "fixed",
    top: 0,
    zIndex: 1000,
    boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)",
  },
  logo: {    
    color: "white",
    height: "60px",
  },
  navLinks: {
    display: "flex", 
    gap: "1.5rem", 
    marginLeft: "2rem", 
  },
  link: {
    color: "white",
    textDecoration: "none",
    fontSize: "1.1rem",
  },
};

export default Navbar;

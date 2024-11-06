import React from "react";
import ImageUpload from "./ImageUpload";
import Navbar from "./Navbar";
import Footer from "./Footer";
import About from "./About";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

function App() {
  return(    
    <>  
      <Navbar />
      <br />
      <br />
      <br />
      <br />
      <Routes>
        <Route path="/" element={<ImageUpload />} />
        <Route path="/about" element={<About />} />
      </Routes>
      <br />
      <br />
      <br />
      <br />
      <br />      
      <Footer />
     
</>
   
  )
}

export default App;

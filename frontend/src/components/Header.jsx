import React from 'react';

const Header = () => {
  return (
    <header className="bg-gradient-to-r from-blue-900 to-blue-700 text-white shadow-lg">
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center space-x-4">
          <img
            src="/image.png"
            alt="Thapar Logo"
            className="h-16 w-16 object-contain"
          />
          <div className="hidden sm:block">
            <p className="text-xs text-blue-200">Thapar Institute of Engineering & Technology</p>
          </div>
        </div>

        {/* Title */}
        <div className="text-center flex-1">
          <h1 className="text-2xl sm:text-3xl font-bold">
            MultiModal Emotion Detection System
          </h1>
          <p className="text-sm text-blue-200 mt-1">Text & Facial Analysis</p>
        </div>

        {/* Right spacer for symmetry */}
        <div className="w-16 sm:w-32"></div>
      </div>
    </header>
  );
};

export default Header;
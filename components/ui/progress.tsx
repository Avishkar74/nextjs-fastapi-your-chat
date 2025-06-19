import React from 'react';

interface ProgressProps {
  value: number; // 0-100
  className?: string;
  showPercentage?: boolean;
}

export const Progress: React.FC<ProgressProps> = ({ 
  value, 
  className = '', 
  showPercentage = false 
}) => {
  const clampedValue = Math.min(Math.max(value, 0), 100);
  
  return (
    <div className={`w-full ${className}`}>
      <div className="flex justify-between items-center mb-1">
        {showPercentage && (
          <span className="text-sm font-medium text-gray-700">{clampedValue}%</span>
        )}
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div 
          className="bg-blue-600 h-2 rounded-full transition-all duration-300 ease-out"
          style={{ width: `${clampedValue}%` }}
        />
      </div>
    </div>
  );
};

interface AnimatedProgressProps {
  isActive: boolean;
  className?: string;
}

export const AnimatedProgress: React.FC<AnimatedProgressProps> = ({ 
  isActive, 
  className = '' 
}) => {
  return (
    <div className={`w-full ${className}`}>
      <div className="w-full bg-gray-200 rounded-full h-1 overflow-hidden">
        <div 
          className={`h-full bg-gradient-to-r from-blue-400 to-blue-600 rounded-full transition-all duration-1000 ${
            isActive ? 'animate-pulse' : ''
          }`}
          style={{
            width: isActive ? '100%' : '0%',
            transform: isActive ? 'translateX(0)' : 'translateX(-100%)',
          }}
        />
      </div>
    </div>
  );
};

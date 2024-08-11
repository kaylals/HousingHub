import React, { useEffect, useRef } from 'react';

declare global {
  interface Window {
    tableau: any;
  }
}

const TableauEmbed: React.FC = () => {
  const ref = useRef<HTMLDivElement>(null);
  const vizRef = useRef<any>(null);
  const url = "https://public.tableau.com/views/HousingHub_Tableau/AverageSales";

  const initViz = () => {
    if (vizRef.current) {
      vizRef.current.dispose();
    }
    vizRef.current = new window.tableau.Viz(ref.current, url);
  }

  useEffect(() => {
    if (ref.current) {
      initViz();
    }
    
    return () => {
      if (vizRef.current) {
        vizRef.current.dispose();
      }
    };
  }, [url]);

  return (
    <div ref={ref} style={{ width: '100vw', height: '100vh' }}></div>
  );
};

export default TableauEmbed;
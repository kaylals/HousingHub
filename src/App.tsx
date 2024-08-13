import React, { useState, useEffect } from 'react';
import './App.css';
import { Grid, Paper, styled, Button, TextField } from '@mui/material';
import CustomDropDown from './component/UI/CustomDropDown';
import TableauEmbed from './component/UI/TableauEmbed';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import dayjs, { Dayjs } from 'dayjs';

const StyledContainer = styled(Grid)({
  padding: 16,
  flex: 1,
  overflow: 'auto',
});

const StyledPaper = styled(Paper)({
  padding: 16,
  textAlign: 'center',
  color: 'text.secondary',
  height: '100%'
});

type Feature = {
  startDate: Dayjs | null;
  endDate: Dayjs | null;
  type: string;
  bedrooms: number;
  bathrooms: number;
};

const App: React.FC = () => {
  const [features, setFeatures] = useState<Feature>({
    startDate: dayjs(),
    endDate: dayjs(),
    type: "",
    bedrooms: 0,
    bathrooms: 0
  });
  const [url, setUrl] = useState<string | null>(null);

  const handleDateChange = (dateType: 'startDate' | 'endDate', newValue: Dayjs | null) => {
    setFeatures((prevFeatures) => ({
      ...prevFeatures,
      [dateType]: newValue,
    }));
  };

  const types = [
    { value: "CONDO", label: 'condo' },
    { value: "RESI", label: 'resi' }
  ];

  const rooms = [];
  for (let i = 1; i <= 5; i++) {
    rooms.push({ value: i, label: i });
  }

  const handleSearch = () => {
    const api = `http://127.0.0.1:5000/xgboost`;
    // /xgboost
    // /random-forest-forecast
    fetch(api)
      .then(response => response.blob()) // 将响应转换为 Blob 对象
      .then(blob => {
        const imageUrl = URL.createObjectURL(blob); // 创建一个对象URL用于显示图片
        setUrl(imageUrl); // 假设 setMovies 用来存储图片URL
      })
      .catch(error => {
        console.error('Error:', error);
      });
  };

  const handleSubmit = () => {
    const requestData = {
      startDate: features.startDate?.format('YYYY-MM-DD'),
      endDate: features.endDate?.format('YYYY-MM-DD'),
      range: 0,
      type: features.type,
      bedrooms: features.bedrooms,
      bathrooms: features.bathrooms,
    };
    const {startDate, endDate} = requestData;
    if (startDate && endDate) {
      const startDateObj = dayjs(startDate);
      const endDateObj = dayjs(endDate);    
      
      const daysDifference = endDateObj.diff(startDateObj, 'day');
      requestData.range = daysDifference;
    }
    console.log(requestData);
    const api = `http://127.0.0.1:5000/xgboost`;
    fetch(api, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData) // Ensure the key matches what your backend expects
      })
      .then(response => response.blob()) // 将响应转换为 Blob 对象
      .then(blob => {
        const imageUrl = URL.createObjectURL(blob); // 创建一个对象URL用于显示图片
        setUrl(imageUrl); // 假设 setMovies 用来存储图片URL
      })
      .catch(error => {
        console.error('Error:', error);
      });
  }

  return (
    <StyledContainer container spacing={3}>
      <Grid item xs={12}>
        <StyledPaper>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={4} md={2}>
              <LocalizationProvider dateAdapter={AdapterDayjs}>
                <DatePicker
                  label="Start Date"
                  value={features.startDate}
                  onChange={(newValue) => handleDateChange('startDate', newValue)}
                />
              </LocalizationProvider>
            </Grid>
            <Grid item xs={4} md={2}>
              <LocalizationProvider dateAdapter={AdapterDayjs}>
                <DatePicker
                  label="End Date"
                  value={features.endDate}
                  onChange={(newValue) => handleDateChange('endDate', newValue)}
                />
              </LocalizationProvider>
            </Grid>
            <Grid item xs={4} md={2}>
              <CustomDropDown
                label="Type"
                value={features.type}
                onChange={(event) => setFeatures({ ...features, type: event.target.value })}
                options={types}
              />
            </Grid>
            <Grid item xs={4} md={2}>
              <CustomDropDown
                label="Bedrooms"
                value={features.bedrooms}
                onChange={(event) => setFeatures({ ...features, bedrooms: parseInt(event.target.value) })}
                options={rooms}
              />
            </Grid>
            <Grid item xs={4} md={2}>
              <CustomDropDown
                label="Bathrooms"
                value={features.bathrooms}
                onChange={(event) => setFeatures({ ...features, bathrooms: parseInt(event.target.value) })}
                options={rooms}
              />
            </Grid>
            <Grid item xs={4} md={2} style={{ textAlign: 'center' }}>
              <Button variant="contained" color="primary" style={{ height: '36px' }} onClick={handleSubmit}>
                Click Me
              </Button>
            </Grid>
          </Grid>
          <Grid container spacing={2} style={{ margin: '5px' }}>
            {url && <img src={url} alt="Fetched PNG" />}
          </Grid>
        </StyledPaper>
      </Grid>
      <Grid item xs={12} style={{ height: '600px', marginTop: '2%' }}>
        <StyledPaper style={{ height: '600px', overflow: 'auto' }}>
          {/* Embed Tableau Dashboard here */}
          <TableauEmbed />
        </StyledPaper>
      </Grid>
    </StyledContainer>
  );
};

export default App;

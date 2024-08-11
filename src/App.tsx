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
  rangeDate: string;
  type: string;
  bedrooms: number;
  bathrooms: number;
};

const App: React.FC = () => {
  const api = `http://127.0.0.1:5000/model1`;
  const [features, setFeatures] = useState<Feature>({
    startDate: dayjs(),
    rangeDate: "",
    type: "",
    bedrooms: 0,
    bathrooms: 0
  });
  const [url, setUrl] = useState<string | null>(null);

  const handleDateChange = (newValue: Dayjs | null) => {
    setFeatures((prevFeatures) => ({
      ...prevFeatures,
      startDate: newValue,
    }));
  };

  const types = [
    { value: "Condo", label: 'condo' },
    { value: "Residential", label: 'resi' }
  ];

  const rooms = [];
  for (let i = 1; i <= 5; i++) {
    rooms.push({ value: i, label: i });
  }

  const handleSearch = () => {
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
      startDate: features.startDate,
      rangeDate: features.rangeDate,
      type: features.type,
      bedrooms: features.bedrooms,
      bathrooms: features.bathrooms,
    };
    console.log(requestData);
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
                  onChange={handleDateChange}
                />
              </LocalizationProvider>
            </Grid>
            <Grid item xs={4} md={2}>
              <TextField
                label="Range of Date"
                value={features.rangeDate}
                onChange={(event) => setFeatures({ ...features, rangeDate: event.target.value })}
                fullWidth
              />
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
              {/* {url && <img src={url} alt="Fetched PNG" />} */}
            </Grid>
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
